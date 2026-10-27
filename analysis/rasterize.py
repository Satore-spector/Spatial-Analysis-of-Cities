"""Step 2: turn each city's .gpkg files into a stack of grids ("channels").

Reads, from data/ (written by data_extraction_osm.R):
    <slug>_boundary.gpkg, <slug>_pois.gpkg, <slug>_buildings.gpkg, <slug>_roads.gpkg
and, if present (written by ghsl_builtup_capitals.R):
    data/ghsl/<slug>_ghs_built_s_2018_10m.tif

Writes, to data/rasters/:
    <slug>.npz          the grid stack (channels x rows x cols), boundary mask, metadata
    <slug>.tif          the same stack as a multi-band GeoTIFF, to open in QGIS
    <slug>_preview.png  quick look: POI density with the detected centre marked
    rasterize_summary.csv

Each cell is --cell metres square (default 100 m) in the city's local UTM
projection, so cells have the same size in every city. Channels: the 10 POI
categories from amenity_wrap.R (count per cell), building footprint share,
major and minor road length, and GHSL built-up share (zeros if no GHSL file).

How it works, in three stages:
  1. prepare    for each city, turn every POI, building and road piece into a
                row of a table: (city, layer, x, y, weight) in metres
  2. aggregate  find the grid cell of every row and add up the weights per
                (city, layer, cell). This is the heavy part: millions of rows.
                --engine pandas  does it in numpy, one city at a time
                --engine spark   does it in Spark SQL, all cities in one
                                 parallel job, reading the table as Parquet
                Both give the same grids.
  3. finish     add the boundary mask and GHSL layer, find the centre, save

Usage (from the project root):
    python -m analysis.rasterize                       all cities found in data/
    python -m analysis.rasterize bengaluru "Pune, India"
    python -m analysis.rasterize --engine spark --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

from .common import (CHANNELS, CHANNEL_UNITS, DATA_DIR, MAJOR_ROADS, channel_name,
                     find_cities, poi_category, raster_dir, save_city_grid, say)

CH_INDEX = {c: i for i, c in enumerate(CHANNELS)}


# ---- Stage 1: prepare -------------------------------------------------------


def _rows(channel, x, y, w) -> pd.DataFrame:
    n = len(x)
    return pd.DataFrame({"channel": np.full(n, CH_INDEX[channel], dtype=np.int16),
                         "x": np.asarray(x, dtype=np.float64), "y": np.asarray(y, dtype=np.float64),
                         "w": np.asarray(w, dtype=np.float64)})


def road_pieces(roads: gpd.GeoDataFrame, cell: float) -> pd.DataFrame:
    """Cut roads into short pieces (a quarter of a cell at most).

    Each piece becomes one row at its midpoint, weighted by its length, so a
    cell's total is the length of road inside it, to within a few per cent.
    """
    if roads.empty:
        return _rows("roads_major", [], [], [])
    lines = roads.explode(index_parts=False)
    lines = lines[lines.geom_type == "LineString"]
    is_major = lines["highway"].isin(MAJOR_ROADS).to_numpy() if "highway" in lines else \
        np.zeros(len(lines), bool)
    geoms = shapely.segmentize(lines.geometry.to_numpy(), cell / 4)
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    same_line = idx[1:] == idx[:-1]
    a, b = coords[:-1][same_line], coords[1:][same_line]
    length = np.hypot(*(b - a).T)
    mid = (a + b) / 2
    major = is_major[idx[:-1][same_line]]
    return pd.concat([_rows("roads_major", mid[major, 0], mid[major, 1], length[major]),
                      _rows("roads_minor", mid[~major, 0], mid[~major, 1], length[~major])])


def prepare_city(slug: str, data_dir: Path, cell: float, poi_source: str) -> dict:
    """Read one city and turn it into a table of weighted points plus its grid."""
    boundary = gpd.read_file(data_dir / f"{slug}_boundary.gpkg")
    utm = boundary.estimate_utm_crs()
    boundary = boundary.to_crs(utm)
    minx, miny, maxx, maxy = boundary.total_bounds
    x0, y1 = np.floor(minx / cell) * cell, np.ceil(maxy / cell) * cell
    ncols = int(np.ceil((maxx - x0) / cell))
    nrows = int(np.ceil((y1 - miny) / cell))
    transform = from_origin(x0, y1, cell, cell)
    say(f"  {slug}: grid {nrows} x {ncols} cells of {cell:g} m ({utm.to_string()})")
    mask = features.rasterize([(g, 1) for g in boundary.geometry], out_shape=(nrows, ncols),
                              transform=transform, fill=0, dtype="uint8").astype(bool)

    # POIs: weight 1, in their category's layer
    pois = gpd.read_file(data_dir / f"{slug}_pois.gpkg").to_crs(utm)
    if poi_source != "both" and "poi_source" in pois:
        pois = pois[pois["poi_source"] == poi_source]
    pois = pois[pois.geometry.notna() & ~pois.geometry.is_empty]
    cats = poi_category(pois)
    pts = pois.geometry.representative_point()
    parts = [_rows(channel_name(cat), pts.x[(cats == cat).to_numpy()], pts.y[(cats == cat).to_numpy()],
                   np.ones(int((cats == cat).sum()))) for cat in cats.unique()]

    # Buildings: footprint area at the building's centre point
    bld = gpd.read_file(data_dir / f"{slug}_buildings.gpkg", columns=[]).to_crs(utm)
    bld = bld[bld.geometry.notna() & ~bld.geometry.is_empty]
    bp = bld.geometry.representative_point()
    parts.append(_rows("buildings", bp.x, bp.y, bld.geometry.area))

    # Roads
    roads = gpd.read_file(data_dir / f"{slug}_roads.gpkg").to_crs(utm)
    parts.append(road_pieces(roads, cell))

    points = pd.concat(parts, ignore_index=True)
    return {"slug": slug, "utm": utm, "transform": transform, "mask": mask,
            "x0": float(x0), "y1": float(y1), "cell": float(cell), "nrows": nrows, "ncols": ncols,
            "points": points, "pois_in_file": len(pois), "buildings": len(bld)}


# ---- Stage 2: aggregate -----------------------------------------------------


def aggregate_numpy(prep: dict) -> np.ndarray:
    """Sum weights per (layer, cell) for one city with numpy."""
    p = prep["points"]
    col = np.floor((p.x.to_numpy() - prep["x0"]) / prep["cell"]).astype(np.int64)
    row = np.floor((prep["y1"] - p.y.to_numpy()) / prep["cell"]).astype(np.int64)
    ok = (row >= 0) & (row < prep["nrows"]) & (col >= 0) & (col < prep["ncols"])
    grid = np.zeros((len(CHANNELS), prep["nrows"], prep["ncols"]))
    np.add.at(grid, (p.channel.to_numpy()[ok], row[ok], col[ok]), p.w.to_numpy()[ok])
    return grid


def aggregate_spark(preps: list[dict], staging: Path) -> dict[str, np.ndarray]:
    """Sum weights per (city, layer, cell) for all cities in one Spark job.

    The point tables are written as one Parquet dataset partitioned by city
    (staging/points/city=<slug>/), the grid definitions as a small table, and
    Spark does:  join -> compute cell row/col -> drop points off the grid ->
    group by (city, layer, row, col) -> sum.
    """
    from pyspark.sql import functions as F

    from .spark_utils import get_spark

    if staging.exists():
        shutil.rmtree(staging)
    (staging / "points").mkdir(parents=True)
    for p in preps:
        d = staging / "points" / f"city={p['slug']}"
        d.mkdir()
        p["points"].to_parquet(d / "part-0.parquet", index=False)
    grids = pd.DataFrame([{k: p[k] for k in ("slug", "x0", "y1", "cell", "nrows", "ncols")} for p in preps])
    grids = grids.rename(columns={"slug": "city"})
    grids.to_parquet(staging / "grids.parquet", index=False)
    n_rows = sum(len(p["points"]) for p in preps)
    say(f"  Spark: aggregating {n_rows:,} rows from {len(preps)} cities")

    spark = get_spark()
    pts = spark.read.parquet(str(staging / "points"))
    grid = spark.read.parquet(str(staging / "grids.parquet"))
    cells = (pts.join(F.broadcast(grid), on="city")
             .withColumn("col", F.floor((F.col("x") - F.col("x0")) / F.col("cell")).cast("long"))
             .withColumn("row", F.floor((F.col("y1") - F.col("y")) / F.col("cell")).cast("long"))
             .where((F.col("row") >= 0) & (F.col("row") < F.col("nrows")) &
                    (F.col("col") >= 0) & (F.col("col") < F.col("ncols")))
             .groupBy("city", "channel", "row", "col")
             .agg(F.sum("w").alias("w")))
    res = cells.toPandas()
    say(f"  Spark: {len(res):,} non-empty (city, layer, cell) totals")

    out = {}
    for p in preps:
        r = res[res.city == p["slug"]]
        g = np.zeros((len(CHANNELS), p["nrows"], p["ncols"]))
        g[r.channel.to_numpy(np.int64), r.row.to_numpy(np.int64), r.col.to_numpy(np.int64)] = r.w.to_numpy()
        out[p["slug"]] = g
    shutil.rmtree(staging, ignore_errors=True)
    return out


# ---- Stage 3: finish --------------------------------------------------------


def ghsl_grid(tif: Path, crs, transform, shape) -> np.ndarray:
    """GHSL built-up surface (m2 per 10 m cell) averaged onto our grid, as a 0-1 share."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(tif) as src:
        src_nodata = src.nodata if src.nodata is not None else np.nan
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_nodata=src_nodata, dst_nodata=np.nan,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
    return np.nan_to_num(out / 100.0, nan=0.0).clip(0, 1)


def find_centre(total: np.ndarray, mask: np.ndarray, cell: float, sigma_m: float = 1000):
    """The busiest place: peak of POI density smoothed over about 1 km.

    Used as the 'CBD' by the classical tests and as the centre of the
    fixed-size window in city mode. Falls back to the middle of the boundary
    if the city has no POIs.
    """
    smooth = ndimage.gaussian_filter(total.astype(float), sigma=sigma_m / cell)
    smooth[~mask] = -1
    if smooth.max() <= 0:
        rows, cols = np.nonzero(mask)
        return int(np.median(rows)), int(np.median(cols))
    r, c = np.unravel_index(np.argmax(smooth), smooth.shape)
    return int(r), int(c)


def finish_city(prep: dict, grid: np.ndarray, data_dir: Path, poi_source: str, engine: str) -> dict:
    slug, cell, mask, utm, transform = prep["slug"], prep["cell"], prep["mask"], prep["utm"], prep["transform"]
    grid = grid.copy()
    b = CH_INDEX["buildings"]
    grid[b] = np.clip(grid[b] / cell ** 2, 0, 1)

    tifs = sorted((data_dir / "ghsl").glob(f"{slug}_ghs_built_s_*.tif"))
    has_ghsl = bool(tifs)
    if has_ghsl:
        grid[CH_INDEX["built_up"]] = ghsl_grid(tifs[0], utm, transform, mask.shape)

    grid = grid.astype(np.float32)
    grid[:, ~mask] = 0  # keep only what is inside the boundary
    poi_idx = [i for i, c in enumerate(CHANNELS) if c.startswith("poi_")]
    total_pois = grid[poi_idx].sum(0)
    centre = find_centre(total_pois, mask, cell)
    cx, cy = prep["x0"] + (centre[1] + 0.5) * cell, prep["y1"] - (centre[0] + 0.5) * cell
    lon, lat = gpd.GeoSeries(gpd.points_from_xy([cx], [cy]), crs=utm).to_crs(4326).iloc[0].coords[0]

    meta = {
        "slug": slug, "channels": CHANNELS, "units": CHANNEL_UNITS, "cell_m": cell,
        "crs": utm.to_wkt(), "epsg": utm.to_epsg(), "transform": list(transform)[:6],
        "centre_rc": list(centre), "centre_lonlat": [lon, lat], "has_ghsl": has_ghsl,
        "poi_source": poi_source, "engine": engine,
    }
    out = raster_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_city_grid(out / f"{slug}.npz", grid, mask, meta)
    write_geotiff(out / f"{slug}.tif", grid, utm, transform)
    preview(out / f"{slug}_preview.png", total_pois, mask, centre, slug, cell)

    roads_km = (grid[CH_INDEX["roads_major"]].sum() + grid[CH_INDEX["roads_minor"]].sum()) / 1000
    return {
        "city": slug, "status": "ok", "engine": engine, "rows": prep["nrows"], "cols": prep["ncols"],
        "cell_m": cell, "area_km2": round(mask.sum() * cell ** 2 / 1e6, 1),
        "pois": int(round(total_pois.sum())), "pois_in_file": prep["pois_in_file"],
        "buildings": prep["buildings"], "road_km": round(float(roads_km), 1), "has_ghsl": has_ghsl,
        "centre_lon": round(lon, 5), "centre_lat": round(lat, 5), "message": "",
    }


def write_geotiff(path, grid, crs, transform):
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=grid.shape[1], width=grid.shape[2],
                       count=grid.shape[0], dtype="float32", crs=crs, transform=transform,
                       compress="deflate") as dst:
        dst.write(grid)
        for i, name in enumerate(CHANNELS, start=1):
            dst.set_band_description(i, name)


def preview(path, total, mask, centre, slug, cell):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.log1p(np.where(mask, total, np.nan))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="magma", interpolation="nearest")
    ax.contour(mask, levels=[0.5], colors="grey", linewidths=0.6)
    ax.plot(centre[1], centre[0], marker="+", color="cyan", markersize=14, mew=2)
    ax.set_title(f"{slug}: POIs per {cell:g} m cell (log), + = detected centre", fontsize=9)
    ax.axis("off")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---- Run --------------------------------------------------------------------


def run(data_dir, cities=None, cell=100.0, poi_source="both", force=False, engine="pandas"):
    data_dir = Path(data_dir)
    slugs = find_cities(data_dir, cities)
    if not slugs:
        raise SystemExit(f"No cities found in {data_dir}. Run data_extraction_osm.R first.")
    rows, preps = [], []
    say(f"Preparing {len(slugs)} cities")
    for slug in slugs:
        if not force and (raster_dir(data_dir) / f"{slug}.npz").exists():
            say(f"  {slug}: grid exists, skipping (use --force to redo)")
            rows.append({"city": slug, "status": "skipped"})
            continue
        try:
            preps.append(prepare_city(slug, data_dir, cell, poi_source))
        except Exception as e:  # one bad city must not stop the rest
            say(f"  {slug}: FAILED: {e}")
            rows.append({"city": slug, "status": "error", "message": str(e)})

    if preps:
        say(f"Aggregating with {engine}")
        if engine == "spark":
            grids = aggregate_spark(preps, raster_dir(data_dir) / "_spark_staging")
        else:
            grids = {p["slug"]: aggregate_numpy(p) for p in preps}
        for p in preps:
            try:
                rows.append(finish_city(p, grids[p["slug"]], data_dir, poi_source, engine))
            except Exception as e:
                say(f"  {p['slug']}: FAILED: {e}")
                rows.append({"city": p["slug"], "status": "error", "message": str(e)})

    summary = pd.DataFrame(rows)
    say("\n" + summary.to_string(index=False))
    raster_dir(data_dir).mkdir(parents=True, exist_ok=True)
    summary.to_csv(raster_dir(data_dir) / "rasterize_summary.csv", index=False)
    if (summary["status"] == "error").any():
        raise SystemExit(1)
    return summary


def add_args(ap):
    ap.add_argument("--cell", type=float, default=100, help="cell size in metres (default 100)")
    ap.add_argument("--poi-source", choices=["both", "node", "area"], default="both",
                    help="which POIs to use; 'node' avoids places counted twice")
    ap.add_argument("--force", action="store_true", help="redo cities that already have a grid")
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas",
                    help="spark: count all cities in one Spark SQL job (needs Java + pyspark)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cities", nargs="*", help="city names or slugs (default: all in data/)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    add_args(ap)
    a = ap.parse_args(argv)
    run(a.data_dir, a.cities, a.cell, a.poi_source, a.force, a.engine)


if __name__ == "__main__":
    main()
