#!/usr/bin/env python3
"""Build the list of US urban centres above a population cutoff.

Source: GHSL Urban Centre Database R2024A (GHS-UCDB), European Commission JRC,
CC BY 4.0. Each urban centre is a polygon defined on a 1 km grid by population
and built-up-share cutoffs, so the list is consistent across the whole country
and needs no name lookups in OpenStreetMap.

Usage (from the project root, inside the project venv):
    python data_extraction/build_city_list.py
    python data_extraction/build_city_list.py --min-pop 100000
    python data_extraction/build_city_list.py --ucdb path/to/GHS_UCDB_GLOBE_R2024A.gpkg
    python data_extraction/build_city_list.py --country "Canada"

Output:
    data/uc/us_cities.csv            one row per city (slug, name, state, population, ...)
    data/uc/us_urban_centres.gpkg    all selected polygons in one file (EPSG:4326)
    data/<slug>_boundary.gpkg        one boundary per city, in the file name the
                                     Python analysis expects (EPSG:4326)
    data/boundaries/<slug>.gpkg      the same polygon, where data_extraction_osm.R
                                     looks for a custom boundary (so OSM extraction
                                     uses this footprint, not a place-name search)

Slugs follow the project's rule (letters only, lower case) and add the state
abbreviation so that the many cities with the same name stay apart, for example
"Portland OR, USA" -> portlandor.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# The global GeoPackage is a ~1.7 GB zip. If this address ever changes, download
# it from https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php and
# pass the file with --ucdb.
UCDB_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/"
    "GHS_UCDB_GLOBE_R2024A/V1-0/GHS_UCDB_GLOBE_R2024A_V1_0.zip"
)
# US state outlines, used only to add a state abbreviation to each city name.
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"

US_NAMES = {"united states", "united states of america", "usa", "us"}


def say(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"Downloading {url}")

    def hook(blocks, block_size, total):
        if total > 0 and blocks % 2000 == 0:
            say(f"  {min(blocks * block_size, total) / 1e6:,.0f} / {total / 1e6:,.0f} MB")

    urllib.request.urlretrieve(url, dest, hook)
    return dest


def get_ucdb(path: Path | None, cache: Path) -> Path:
    """Return the path to the UCDB GeoPackage, downloading and unzipping if needed."""
    if path:
        path = Path(path)
        if path.suffix.lower() == ".zip":
            return unzip_gpkg(path, cache)
        return path
    found = sorted(cache.glob("**/*UCDB*.gpkg"))
    if found:
        say(f"Using cached {found[0]}")
        return found[0]
    zpath = download(UCDB_URL, cache / "GHS_UCDB_GLOBE_R2024A_V1_0.zip")
    return unzip_gpkg(zpath, cache)


def unzip_gpkg(zpath: Path, cache: Path) -> Path:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".gpkg")]
        if not names:
            sys.exit(f"No .gpkg inside {zpath}. Contents: {z.namelist()[:10]}")
        say(f"Extracting {names[0]}")
        z.extract(names[0], cache)
    return cache / names[0]


def resolve(cols, prefix: str, year: int | None):
    """Pick the column starting with `prefix`, preferring `year`, else the latest year."""
    cands = [c for c in cols if c.upper().startswith(prefix)]
    if not cands:
        return None
    if year:
        for c in cands:
            if c.endswith(str(year)):
                return c
    return sorted(cands)[-1]


def load_ucdb(gpkg: Path, year: int | None) -> gpd.GeoDataFrame:
    """Read the UCDB and return geometry + id, name, country, population columns.

    The columns are found by prefix (GC_UCN_MAI_*, GC_CNT_GAD_*, GC_POP_TOT_*) so a
    small change in the year suffix does not break the script. If the attributes are
    spread over several layers they are joined on ID_UC_G0.
    """
    import pyogrio

    layers = [row[0] for row in pyogrio.list_layers(gpkg)]
    say(f"Layers in {gpkg.name}: {layers}")
    wanted = {"name": "GC_UCN_MAI_", "country": "GC_CNT_GAD_", "pop": "GC_POP_TOT_"}
    geom_layer, frames, picked = None, {}, {}
    for lyr in layers:
        head = gpd.read_file(gpkg, layer=lyr, rows=1)
        if geom_layer is None and head.geometry.notna().any() and \
                head.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon"):
            geom_layer = lyr
        frames[lyr] = list(head.columns)
        for key, prefix in wanted.items():
            col = resolve(head.columns, prefix, year)
            if col and key not in picked:
                picked[key] = (lyr, col)
    missing = [k for k in wanted if k not in picked]
    if geom_layer is None or missing:
        for lyr, cols in frames.items():
            say(f"  {lyr}: {cols[:40]}")
        sys.exit(f"Could not find the polygon layer or columns {missing}. "
                 "Check the column names above and adjust `wanted` in this script.")

    gdf = gpd.read_file(gpkg, layer=geom_layer)
    idcol = next((c for c in gdf.columns if c.upper() == "ID_UC_G0"), None)
    out = gdf[[idcol, "geometry"]].copy() if idcol else gdf[["geometry"]].copy()
    for key, (lyr, col) in picked.items():
        if lyr == geom_layer:
            out[key] = gdf[col].values
        else:
            if not idcol:
                sys.exit(f"{col} is in layer {lyr}, but there is no ID_UC_G0 to join on.")
            other = gpd.read_file(gpkg, layer=lyr, ignore_geometry=True)[[idcol, col]]
            out = out.merge(other.rename(columns={col: key}), on=idcol, how="left")
    out = out.rename(columns={idcol: "ucid"}) if idcol else out.assign(ucid=range(len(out)))
    say("Columns used: " + ", ".join(f"{k}={c}" for k, (_, c) in picked.items()))
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def add_state(gdf: gpd.GeoDataFrame, cache: Path) -> gpd.GeoDataFrame:
    """Add a `state` abbreviation from the centroid of each city, if the outlines download."""
    gdf = gdf.copy()
    gdf["state"] = ""
    try:
        zpath = cache / "us_states.zip"
        if not zpath.exists():
            urllib.request.urlretrieve(STATES_URL, zpath)
        states = gpd.read_file(f"zip://{zpath}").to_crs(4326)
        col = "STUSPS" if "STUSPS" in states.columns else states.columns[0]
        cent = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=4326)
        joined = gpd.sjoin(cent, states[[col, "geometry"]], how="left", predicate="within")
        joined = joined[~joined.index.duplicated()]
        gdf["state"] = joined[col].fillna("").astype(str).values
        # centroids just offshore (coastal cities): take the nearest state instead
        for i in gdf.index[gdf["state"] == ""]:
            p = cent.geometry.loc[i]
            gdf.loc[i, "state"] = states.iloc[states.distance(p).values.argmin()][col]
    except Exception as e:  # no network, changed address, ... -> numbered suffixes instead
        say(f"  (state lookup skipped: {e}); duplicate names get a letter suffix instead")
    return gdf


def make_slugs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Unique, letters-only slugs that follow the project's city_slug() rule."""
    gdf = gdf.copy()
    base = (gdf["name"].astype(str) + gdf["state"].astype(str)).str.lower().str.replace(
        r"[^a-z]", "", regex=True)
    base = base.where(base != "", "city")
    slugs, seen = [], {}
    for b in base:  # gdf is sorted by population, so the biggest keeps the plain name
        n = seen.get(b, 0)
        seen[b] = n + 1
        slugs.append(b if n == 0 else b + _letters(n))
    gdf["slug"] = slugs
    st = gdf["state"].replace("", pd.NA)
    gdf["city"] = [
        f"{n} {s}, USA" if isinstance(s, str) else f"{n}, USA" for n, s in zip(gdf["name"], st)
    ]
    return gdf


def _letters(n: int) -> str:
    """1 -> 'a', 2 -> 'b', ... 27 -> 'aa' (digits would be dropped by city_slug)."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(97 + r) + s
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucdb", type=Path, help="GHS-UCDB .gpkg or .zip you already downloaded")
    ap.add_argument("--country", default="United States", help="country name as in the UCDB")
    ap.add_argument("--min-pop", type=float, default=50_000, help="keep centres with MORE than this many people")
    ap.add_argument("--year", type=int, default=2025, help="population/name year in the UCDB columns")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--no-boundaries", action="store_true", help="only write the list, not per-city boundary files")
    ap.add_argument("--force", action="store_true", help="overwrite existing boundary files")
    args = ap.parse_args()

    cache = args.data_dir / "uc"
    cache.mkdir(parents=True, exist_ok=True)
    gdf = load_ucdb(get_ucdb(args.ucdb, cache), args.year)
    say(f"{len(gdf):,} urban centres worldwide")

    want = {args.country.lower()} | (US_NAMES if args.country.lower() in US_NAMES else set())
    sel = gdf[gdf["country"].astype(str).str.lower().isin(want)]
    if sel.empty:
        like = sorted({c for c in gdf["country"].dropna().astype(str)
                       if args.country.lower()[:4] in c.lower()})
        sys.exit(f"No urban centres for country '{args.country}'. Similar values: {like[:10]}")
    sel = sel[pd.to_numeric(sel["pop"], errors="coerce") > args.min_pop]
    sel = sel.sort_values("pop", ascending=False).reset_index(drop=True)
    say(f"{len(sel):,} urban centres in {args.country} with population > {args.min_pop:,.0f}")

    sel["area_km2"] = (sel.to_crs("EPSG:6933").area / 1e6).round(1)
    sel = sel.to_crs(4326)
    if args.country.lower() in US_NAMES:
        sel = add_state(sel, cache)
    else:
        sel["state"] = ""
    sel = make_slugs(sel)
    c = sel.geometry.representative_point()
    sel["lon"], sel["lat"] = c.x.round(5), c.y.round(5)

    cols = ["slug", "city", "name", "state", "pop", "area_km2", "lon", "lat", "ucid"]
    csv = cache / "us_cities.csv"
    sel[cols].to_csv(csv, index=False)
    sel[cols + ["geometry"]].to_file(cache / "us_urban_centres.gpkg", driver="GPKG")
    say(f"Wrote {csv} and {cache / 'us_urban_centres.gpkg'}")

    if not args.no_boundaries:
        # Two copies, because two scripts look in different places:
        #   data/boundaries/<slug>.gpkg   read by data_extraction_osm.R (overrides the place-name search)
        #   data/<slug>_boundary.gpkg     read by ghsl_builtup_100m.py and the Python analysis
        bdir = args.data_dir / "boundaries"
        bdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for row in sel.itertuples():
            for f in (bdir / f"{row.slug}.gpkg", args.data_dir / f"{row.slug}_boundary.gpkg"):
                if f.exists() and not args.force:
                    continue
                gpd.GeoDataFrame({"city": [row.city], "ucid": [row.ucid]},
                                 geometry=[row.geometry], crs=4326).to_file(f, driver="GPKG")
                n += 1
        say(f"Wrote {n} boundary files to {args.data_dir}/ and {bdir}/ (existing ones kept; --force to overwrite)")

    say("\nLargest five:\n" + sel[["city", "pop", "area_km2"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
