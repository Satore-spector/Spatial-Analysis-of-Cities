#!/usr/bin/env python3
"""Extract OpenStreetMap POIs, buildings and roads for hundreds of cities at once.

This replaces data_extraction_osm.R when there are many cities. The R script
reads the whole regional OSM file three times per city (minutes per city);
this script reads each regional file once per layer and hands every feature
to whichever cities it falls in, so the time hardly depends on the number of
cities. The outputs are the same files, in the same format, that the R
script writes and the Python analysis reads:

    <data-dir>/<slug>_pois.gpkg       POIs (points; places mapped as areas become one point each)
    <data-dir>/<slug>_buildings.gpkg  building footprints
    <data-dir>/<slug>_roads.gpkg      roads (motorway ... residential, unclassified, links)
    <data-dir>/osm_extraction_summary.csv

The city boundaries come from <data-dir>/<slug>_boundary.gpkg (written by
build_city_list.py), and the cities from its CSV.

Regional OSM files come from Geofabrik. For India the default is its six zone
files (about 1.6 GB in total). Downloads resume where they stopped and are
checked against Geofabrik's MD5 checksums. Files you already have are reused,
including the ones osmextract (R) saved as geofabrik_<zone>-latest.osm.pbf.

Usage (from the project root, inside the project venv):
    python data_extraction/osm_batch_extract.py --cities-csv data/india/uc/india_cities.csv --data-dir data/india
    ... --zones southern-zone western-zone      only some zones
    ... --pbf path/to/india-latest.osm.pbf      use files you already have, no download
    ... --only bengaluru mysuru                 only some cities (still one pass per file)
    ... --download-only                         just fetch the files

Resumable: each finished (file, layer) pass is recorded in
<data-dir>/osm_staging/, so a rerun after a crash skips the passes already
done. --force starts again.

How a feature is given to a city: POIs and buildings by a point inside them
(the same point the analysis uses), roads if any part of them lies inside the
boundary. Features in two overlapping regional files are kept once.

Reading needs GDAL's OSM driver, which comes with pyogrio. GDAL keeps a
temporary node index on disk while reading (several GB for a large region);
it goes in <pbf-dir>/tmp/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import shapely

GEOFABRIK = "https://download.geofabrik.de"
REGIONS = {
    "india": ("asia/india", ["central-zone", "eastern-zone", "north-eastern-zone",
                             "northern-zone", "southern-zone", "western-zone"]),
}
POI_TAGS = ["amenity", "shop", "office", "public_transport", "railway", "tourism",
            "leisure", "healthcare"]
ROAD_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential",
              "unclassified", "motorway_link", "trunk_link", "primary_link",
              "secondary_link", "tertiary_link"]
LAYERS = {
    "points": {"columns": ["osm_id", "name"] + POI_TAGS,
               "where": " OR ".join(f"{t} IS NOT NULL" for t in POI_TAGS)},
    "multipolygons": {"columns": ["osm_id", "osm_way_id", "name", "building"] + POI_TAGS,
                      "where": " OR ".join(f"{t} IS NOT NULL" for t in ["building"] + POI_TAGS)},
    "lines": {"columns": ["osm_id", "name", "highway"],
              "where": "highway IN (" + ",".join(f"'{t}'" for t in ROAD_TYPES) + ")"},
}
FLUSH_ROWS = 200_000  # per city and kind, kept in memory before writing to disk


def say(msg: str) -> None:
    print(msg, flush=True)


# ---- Downloads ---------------------------------------------------------------


def download(url: str, dest: Path, tries: int = 6) -> Path:
    """Resumable download (HTTP Range) with an MD5 check against url + '.md5'."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored the range: start again
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0) or 0)
                say(f"  downloading {url} ({total / 1e6:,.0f} MB, resuming at {have / 1e6:,.0f} MB)"
                    if have else f"  downloading {url} ({total / 1e6:,.0f} MB)")
                with open(part, "ab" if have else "wb") as f:
                    done, last = have, time.time()
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 15:
                            say(f"    {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                            last = time.time()
            break
        except Exception as e:  # dropped connection: resume from where it stopped
            say(f"    interrupted ({e}); retry {attempt}/{tries}")
            time.sleep(min(60, 5 * attempt))
    else:
        sys.exit(f"Could not download {url}. Rerun to resume, or download it in a browser and "
                 f"save it as {dest}.")
    try:
        with urllib.request.urlopen(url + ".md5", timeout=60) as r:
            want = r.read().decode().split()[0]
        h = hashlib.md5()
        with open(part, "rb") as f:
            while chunk := f.read(1 << 22):
                h.update(chunk)
        if h.hexdigest() != want:
            part.unlink()
            sys.exit(f"{dest.name} is damaged (checksum mismatch) and was deleted. Rerun to download again.")
        say(f"  checksum ok: {dest.name}")
    except urllib.error.URLError:
        say(f"  (no checksum available for {dest.name}; not verified)")
    part.replace(dest)
    return dest


def region_files(region: str, zones: list[str] | None, pbf_dir: Path, download_missing: bool) -> list[Path]:
    path, all_zones = REGIONS[region]
    files = []
    for z in zones or all_zones:
        if z not in all_zones:
            sys.exit(f"Unknown zone '{z}'. {region} zones: {', '.join(all_zones)}")
        mine, from_r = pbf_dir / f"{z}-latest.osm.pbf", pbf_dir / f"geofabrik_{z}-latest.osm.pbf"
        if from_r.exists() and not mine.exists():
            say(f"  reusing {from_r.name} (downloaded earlier by the R script)")
            files.append(from_r)
        elif mine.exists() or not download_missing:
            files.append(mine)
        else:
            files.append(download(f"{GEOFABRIK}/{path}/{z}-latest.osm.pbf", mine))
    return files


# ---- Reading one OSM file -----------------------------------------------------


def write_osmconf(path: Path) -> Path:
    """GDAL's OSM settings with the tags this project needs as their own columns."""
    src = Path(pyogrio.get_gdal_data_path()) / "osmconf.ini"
    out, section = [], None
    for line in src.read_text().splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[]")
        if s.startswith("closed_ways_are_polygons=") and "healthcare" not in s:
            line = line + ",healthcare"
        if s.startswith("attributes=") and section in LAYERS:
            cols = [c for c in LAYERS[section]["columns"] if c not in ("osm_id", "osm_way_id")]
            line = "attributes=" + ",".join(cols)
        if s.startswith("other_tags=") and section in LAYERS:
            line = "other_tags=no"
        out.append(line)
    path.write_text("\n".join(out) + "\n")
    return path


class CitySink:
    """Buffers rows per (city, kind) and appends them to Parquet files on disk."""

    def __init__(self, root: Path, tag: str):
        self.root, self.tag, self.buf, self.n_files = root, tag, {}, 0

    def add(self, kind: str, slug: str, df: pd.DataFrame):
        lst = self.buf.setdefault((kind, slug), [])
        lst.append(df)
        if sum(len(d) for d in lst) >= FLUSH_ROWS:
            self.flush(kind, slug)

    def flush(self, kind=None, slug=None):
        keys = [(kind, slug)] if kind else list(self.buf)
        for k in keys:
            parts = self.buf.pop(k, [])
            if not parts:
                continue
            d = self.root / k[0] / f"city={k[1]}"
            d.mkdir(parents=True, exist_ok=True)
            self.n_files += 1
            pq.write_table(pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False),
                           d / f"{self.tag}-{self.n_files:06d}.parquet")


def stream_layer(pbf: Path, layer: str, batch_size: int):
    """Yield (attributes DataFrame, shapely geometries) for one OSM layer, in batches."""
    spec = LAYERS[layer]
    with pyogrio.open_arrow(pbf, layer=layer, columns=spec["columns"], where=spec["where"],
                            batch_size=batch_size, use_pyarrow=True) as (meta, reader):
        gcol = meta["geometry_name"] or "wkb_geometry"
        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            geoms = shapely.from_wkb(tbl.column(gcol).to_numpy(zero_copy_only=False))
            df = tbl.drop([gcol]).to_pandas()
            for c in spec["columns"]:
                if c not in df:
                    df[c] = None
            yield df, geoms


def run_pass(pbf: Path, layer: str, cities: gpd.GeoDataFrame, sink: CitySink, batch_size: int) -> dict:
    """One read of one layer of one OSM file; every feature goes to the cities it is in."""
    tree = shapely.STRtree(cities.geometry.values)
    slugs = cities["slug"].to_numpy()
    counts = {"read": 0, "kept": 0}
    t0 = time.time()
    for df, geoms in stream_layer(pbf, layer, batch_size):
        counts["read"] += len(df)
        ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
        if layer == "lines":
            fi, ci = tree.query(geoms, predicate="intersects")
            fi, ci = fi[ok[fi]], ci[ok[fi]]
        else:
            rep = np.full(len(geoms), None, dtype=object)
            rep[ok] = shapely.point_on_surface(shapely.make_valid(geoms[ok])) if layer == "multipolygons" \
                else geoms[ok]
            fi, ci = tree.query(rep, predicate="within")
        if not len(fi):
            continue
        counts["kept"] += len(fi)
        part = df.iloc[fi].reset_index(drop=True)
        part["slug"] = slugs[ci]
        part["wkb"] = shapely.to_wkb(geoms[fi])
        if layer == "multipolygons":
            part["rep_wkb"] = shapely.to_wkb(rep[fi])
        for slug, g in part.groupby("slug", sort=False):
            g = g.drop(columns="slug")
            if layer == "points":
                sink.add("pois_node", slug, g)
            elif layer == "lines":
                sink.add("roads", slug, g)
            else:
                b = g[g["building"].notna()]
                if len(b):
                    sink.add("buildings", slug, b.drop(columns="rep_wkb"))
                a = g[g[POI_TAGS].notna().any(axis=1)]
                if len(a):
                    sink.add("pois_area", slug, a)
        if counts["read"] % (batch_size * 20) < batch_size:
            say(f"    {layer}: {counts['read']:,} read, {counts['kept']:,} kept ({time.time() - t0:,.0f} s)")
    sink.flush()
    counts["seconds"] = round(time.time() - t0)
    return counts


# ---- Writing each city --------------------------------------------------------


def read_kind(staging: Path, kind: str, slug: str) -> pd.DataFrame | None:
    d = staging / kind / f"city={slug}"
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return None
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def finish_city(slug: str, staging: Path, data_dir: Path) -> dict:
    row = {"city": slug, "status": "ok"}
    # POIs: nodes as they are, areas as one point each (the R schema)
    parts = []
    nodes = read_kind(staging, "pois_node", slug)
    if nodes is not None:
        nodes = nodes.drop_duplicates("osm_id")
        nodes["poi_source"] = "node"
        parts.append(gpd.GeoDataFrame(nodes.drop(columns="wkb"), geometry=shapely.from_wkb(nodes.wkb), crs=4326))
    areas = read_kind(staging, "pois_area", slug)
    n_area = 0
    if areas is not None:
        # Relations (osm_id) and ways (osm_way_id) are numbered separately, so the
        # same number can be both: de-duplicate on the type and the number.
        key = ("r" + areas["osm_id"].astype("string")).fillna("w" + areas["osm_way_id"].astype("string"))
        areas = areas.assign(_key=key).drop_duplicates("_key").drop(columns="_key")
        areas["osm_id"] = areas["osm_id"].fillna(areas["osm_way_id"])
        areas["poi_source"] = "area"
        n_area = len(areas)
        parts.append(gpd.GeoDataFrame(areas.drop(columns=["wkb", "rep_wkb", "osm_way_id", "building"]),
                                      geometry=shapely.from_wkb(areas.rep_wkb), crs=4326))
    cols = ["osm_id", "name"] + POI_TAGS + ["poi_source"]
    pois = pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(columns=cols + ["geometry"],
                                                                             geometry="geometry", crs=4326)
    pois = gpd.GeoDataFrame(pois[cols + ["geometry"]], geometry="geometry", crs=4326)
    pois["osm_id"] = pois["osm_id"].astype(str)
    pois.to_file(data_dir / f"{slug}_pois.gpkg", driver="GPKG")

    b = read_kind(staging, "buildings", slug)
    if b is None:
        b = pd.DataFrame(columns=["osm_id", "osm_way_id", "name", "building", "wkb"])
    b = b.assign(key=("r" + b["osm_id"].astype("string")).fillna("w" + b["osm_way_id"].astype("string")))
    b = b.drop_duplicates("key")
    bld = gpd.GeoDataFrame(b[["osm_id", "osm_way_id", "name", "building"]],
                           geometry=shapely.from_wkb(b["wkb"].to_numpy()), crs=4326)
    bld.to_file(data_dir / f"{slug}_buildings.gpkg", driver="GPKG")

    r = read_kind(staging, "roads", slug)
    if r is None:
        r = pd.DataFrame(columns=["osm_id", "name", "highway", "wkb"])
    r = r.drop_duplicates("osm_id")
    roads = gpd.GeoDataFrame(r[["osm_id", "name", "highway"]], geometry=shapely.from_wkb(r["wkb"].to_numpy()),
                             crs=4326)
    roads.to_file(data_dir / f"{slug}_roads.gpkg", driver="GPKG")

    row.update(pois=len(pois), pois_from_areas=n_area, buildings=len(bld), roads=len(roads))
    if len(pois) == 0 and len(bld) == 0:
        row["status"] = "empty"
        row["message"] = "no OSM features inside the boundary (is its region file included?)"
    return row


# ---- Main ---------------------------------------------------------------------


def load_cities(csv: Path, data_dir: Path, only) -> gpd.GeoDataFrame:
    lst = pd.read_csv(csv)
    if only:
        lst = lst[lst["slug"].isin(only)]
    rows, missing = [], []
    for slug in lst["slug"]:
        f = data_dir / f"{slug}_boundary.gpkg"
        if not f.exists():
            missing.append(slug)
            continue
        g = gpd.read_file(f).to_crs(4326)
        rows.append({"slug": slug, "geometry": shapely.make_valid(g.union_all())})
    if missing:
        say(f"  {len(missing)} cities have no boundary file and are skipped: {', '.join(missing[:10])} ...")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities-csv", type=Path, required=True, help="list from build_city_list.py")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--region", default="india", choices=sorted(REGIONS))
    ap.add_argument("--zones", nargs="*", help="only these Geofabrik zones (default: all of the region)")
    ap.add_argument("--pbf", nargs="*", type=Path, help="use these .osm.pbf files instead of downloading")
    ap.add_argument("--pbf-dir", type=Path, default=Path("data/osm_pbf"),
                    help="where regional files are kept (shared by all data dirs)")
    ap.add_argument("--only", nargs="*", help="only these city slugs")
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo every pass and every city")
    args = ap.parse_args(argv)

    data_dir = args.data_dir
    staging = data_dir / "osm_staging"
    if args.force and staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    say("OSM files")
    pbfs = list(args.pbf) if args.pbf else region_files(args.region, args.zones, args.pbf_dir, True)
    missing = [p for p in pbfs if not p.exists()]
    if missing:
        sys.exit(f"Missing OSM file(s): {', '.join(map(str, missing))}")
    if args.download_only:
        say("Downloads done.")
        return 0

    cities = load_cities(args.cities_csv, data_dir, args.only)
    if cities.empty:
        sys.exit("No cities with boundary files. Run build_city_list.py first.")
    say(f"{len(cities)} cities, {len(pbfs)} OSM file(s)")

    tmp = args.pbf_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    pyogrio.set_gdal_config_options({
        "OSM_CONFIG_FILE": str(write_osmconf(staging / "osmconf.ini")),
        "CPL_TMPDIR": str(tmp),
        "OSM_MAX_TMPFILE_SIZE": "1024",       # MB of node index kept in memory before using disk
        "OGR_INTERLEAVED_READING": "YES",
    })
    log = []
    for pbf in pbfs:
        for layer in ("points", "multipolygons", "lines"):
            done = staging / f"done_{pbf.stem}_{layer}.txt"
            if done.exists():
                say(f"  {pbf.name} / {layer}: done earlier, skipping")
                continue
            # Remove half-written output of an interrupted pass of this file and layer.
            for f in staging.glob(f"*/city=*/{pbf.stem}_{layer}-*.parquet"):
                f.unlink()
            say(f"  {pbf.name} / {layer}: reading")
            c = run_pass(pbf, layer, cities, CitySink(staging, f"{pbf.stem}_{layer}"), args.batch_size)
            done.write_text(str(c))
            log.append({"file": pbf.name, "layer": layer, **c})
            say(f"  {pbf.name} / {layer}: {c['read']:,} features read, {c['kept']:,} kept, {c['seconds']:,} s")

    say("Writing city files")
    rows = []
    for slug in cities["slug"]:
        try:
            rows.append(finish_city(slug, staging, data_dir))
        except Exception as e:
            rows.append({"city": slug, "status": "error", "message": str(e)[:300]})
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "osm_extraction_summary.csv", index=False)
    say(summary["status"].value_counts().to_string())
    bad = summary[summary["status"] != "ok"]
    if len(bad):
        say("Not ok:\n" + bad.to_string(index=False))
    say(f"Summary: {data_dir / 'osm_extraction_summary.csv'}")
    return 0 if (summary["status"] != "error").all() else 1


if __name__ == "__main__":
    sys.exit(main())
