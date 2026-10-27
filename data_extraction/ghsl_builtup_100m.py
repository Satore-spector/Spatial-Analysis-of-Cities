#!/usr/bin/env python3
"""Download GHSL built-up surface at 100 m for every city in data/uc/us_cities.csv.

Why 100 m: the Python analysis averages the 10 m GHSL layer onto a 100 m grid
anyway, so the native 100 m product (GHS-BUILT-S, release P2023A) gives the same
information with 100 times fewer pixels. One Earth Engine request per city is
enough (a few MB), instead of dozens of tiles per city.

Why epoch 2020 by default: it is the latest epoch built from observations. The
2025 and 2030 layers in the same collection are projections. Change it with --epoch.

Usage (from the project root, inside the project venv):
    python data_extraction/ghsl_builtup_100m.py --project=my-gcp-project
    python data_extraction/ghsl_builtup_100m.py --project=my-gcp-project --limit 10
    python data_extraction/ghsl_builtup_100m.py --project=my-gcp-project --only newyorkny losangelesca
    (the project can also come from the EE_PROJECT environment variable)

Output, in data/ghsl/:
    <slug>_ghs_built_s_<epoch>_100m.tif   built-up surface as % of each 100 m cell (0-100),
                                           in the city's UTM zone, on a 100 m grid
    ghsl_100m_summary.csv                 one row per city with sanity checks

The values are scaled to 0-100 (the source stores m2 per cell, 0-10000) so that
analysis/rasterize.py reads them unchanged: it divides by 100 to get a 0-1 share.

Resumable: cities whose .tif already exists are skipped (--force to redo).
Data: GHSL, European Commission JRC, CC BY 4.0 (credit the source).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.merge import merge

ASSET = "JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}"
BAND = "built_surface"  # m2 of buildings per 100 m x 100 m cell (0-10000)
SCALE = 100
MAX_SPLIT_DEPTH = 4


def say(msg: str) -> None:
    print(msg, flush=True)


# ---- Earth Engine ------------------------------------------------------------


def init_ee(project: str | None):
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:
        say("Earth Engine needs a login (a browser window will open).")
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def fetch_bytes(ee, image, bbox, epsg: int) -> bytes:
    """One getDownloadURL request for bbox = (minx, miny, maxx, maxy) in lon/lat.

    crs_transform pins the grid to multiples of 100 m in the UTM zone, so tiles
    fetched separately (see fetch_region) line up exactly.
    """
    import urllib.request

    region = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
    url = image.getDownloadURL({
        "region": region, "crs": f"EPSG:{epsg}", "crs_transform": [SCALE, 0, 0, 0, -SCALE, 0],
        "format": "GEO_TIFF",
    })
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    if data[:2] == b"PK":  # sometimes returned zipped
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(next(n for n in z.namelist() if n.lower().endswith(".tif")))
    return data


def too_large(err: Exception) -> bool:
    m = str(err).lower()
    return "must be less than or equal to" in m or "too large" in m or "request size" in m


def fetch_region(fetch, bbox, depth=0) -> bytes:
    """Fetch bbox; if Earth Engine says it is too big, split it in two and mosaic."""
    try:
        return with_retries(lambda: fetch(bbox))
    except Exception as e:
        if not too_large(e) or depth >= MAX_SPLIT_DEPTH:
            raise
    x0, y0, x1, y1 = bbox
    if (x1 - x0) >= (y1 - y0):
        xm = (x0 + x1) / 2
        parts = [(x0, y0, xm, y1), (xm, y0, x1, y1)]
    else:
        ym = (y0 + y1) / 2
        parts = [(x0, y0, x1, ym), (x0, ym, x1, y1)]
    return mosaic([fetch_region(fetch, p, depth + 1) for p in parts])


def with_retries(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if too_large(e) or i == tries - 1:
                raise
            time.sleep(5 * 2 ** i)


def mosaic(pieces: list[bytes]) -> bytes:
    files = [rasterio.MemoryFile(p) for p in pieces]
    try:
        srcs = [f.open() for f in files]
        arr, transform = merge(srcs, nodata=srcs[0].nodata)
        profile = srcs[0].profile
        profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform)
        with rasterio.MemoryFile() as out:
            with out.open(**profile) as dst:
                dst.write(arr)
            return out.read()
    finally:
        for f in files:
            f.close()


# ---- Per city ----------------------------------------------------------------


def process_city(fetch, row, data_dir: Path, epoch: int, force: bool) -> dict:
    slug = row["slug"]
    out = data_dir / "ghsl" / f"{slug}_ghs_built_s_{epoch}_100m.tif"
    bfile = data_dir / f"{slug}_boundary.gpkg"
    if out.exists() and not force:
        return {"slug": slug, "status": "skipped"}
    if not bfile.exists():
        return {"slug": slug, "status": "error", "message": f"missing {bfile.name}"}

    b = gpd.read_file(bfile).to_crs(4326)
    epsg = b.estimate_utm_crs().to_epsg()
    minx, miny, maxx, maxy = b.total_bounds
    pad = 0.01  # ~1 km, so cells on the edge of the boundary are complete
    bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    data = fetch_region(lambda bb: fetch(bb, epsg), bbox)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)

    # sanity numbers: built-up area inside the boundary, and its share of the boundary
    with rasterio.open(out) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        inside = geometry_mask(b.to_crs(src.crs).geometry, out_shape=src.shape,
                               transform=src.transform, invert=True)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    cell_km2 = (SCALE / 1000) ** 2
    vals = arr[inside]
    built_km2 = float(np.nansum(vals)) / 100 * cell_km2
    area_km2 = float(inside.sum()) * cell_km2
    return {
        "slug": slug, "status": "ok", "epoch": epoch, "utm_epsg": epsg,
        "boundary_cells_km2": round(area_km2, 1),
        "built_km2": round(built_km2, 2),
        "built_pct": round(100 * built_km2 / area_km2, 1) if area_km2 else "",
        "nan_cells_in_boundary": int(np.isnan(vals).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                    help="Google Cloud project with Earth Engine enabled (or env EE_PROJECT)")
    ap.add_argument("--cities-csv", type=Path, default=Path("data/uc/us_cities.csv"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--epoch", type=int, default=2020, help="GHSL epoch (1975-2030 in 5-year steps)")
    ap.add_argument("--workers", type=int, default=4, help="parallel Earth Engine requests")
    ap.add_argument("--limit", type=int, help="only the first N cities (largest first)")
    ap.add_argument("--only", nargs="*", help="only these slugs")
    ap.add_argument("--force", action="store_true", help="redownload cities that already have a .tif")
    ap.add_argument("--summary", default="ghsl_100m_summary.csv")
    args = ap.parse_args()

    if not args.project:
        sys.exit("Pass your Google Cloud project: --project=<id> (or set EE_PROJECT).")
    if not args.cities_csv.exists():
        sys.exit(f"{args.cities_csv} not found. Run data_extraction/build_city_list.py first.")

    with open(args.cities_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.only:
        rows = [r for r in rows if r["slug"] in set(args.only)]
    if args.limit:
        rows = rows[: args.limit]
    say(f"{len(rows)} cities, GHSL epoch {args.epoch}, {SCALE} m")

    ee = init_ee(args.project)
    image = ee.Image(ASSET.format(epoch=args.epoch)).select(BAND).divide(100).toFloat()

    def fetch(bbox, epsg):
        return fetch_bytes(ee, image, bbox, epsg)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(process_city, fetch, r, args.data_dir, args.epoch, args.force): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"slug": r["slug"], "status": "error", "message": str(e)[:300]}
            results.append(res)
            extra = f"{res.get('built_km2', '')} km2 built ({res.get('built_pct', '')}%)" if res["status"] == "ok" else res.get("message", "")
            say(f"[{i}/{len(rows)}] {res['slug']}: {res['status']} {extra}")

    out = args.data_dir / "ghsl" / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r}, key=lambda k: (k != "slug", k != "status", k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: r["slug"]))
    bad = [r["slug"] for r in results if r["status"] == "error"]
    say(f"\nSummary: {out}")
    if bad:
        say(f"{len(bad)} failed: {', '.join(bad[:20])}{' ...' if len(bad) > 20 else ''}. Re-run to retry them.")


if __name__ == "__main__":
    main()
