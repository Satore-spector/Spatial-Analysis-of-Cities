"""Export city boundaries from city_types.gpkg to a small GeoJSON for the site.

Run from the project root:
    python export_boundaries.py
Options:
    --tol 0.006     simplification in degrees (bigger = smaller file, rougher shapes)
    --src PATH      city_types.gpkg location
    --out PATH      output GeoJSON location
"""
import argparse
import os
import sys

import geopandas as gpd

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/india/analysis/classification/city_types.gpkg")
p.add_argument("--out", default="site/public/city_boundaries.geojson")
p.add_argument("--tol", type=float, default=0.003)
a = p.parse_args()

if not os.path.exists(a.src):
    sys.exit(f"Not found: {a.src}\nRun from the project root, or pass --src.")

d = gpd.read_file(a.src)
if "city" not in d.columns:
    sys.exit(f"No 'city' column in {a.src}. Columns: {list(d.columns)}")

d = d[["city", "geometry"]].dropna(subset=["geometry"])
d = d[~d.geometry.is_empty]
if d.crs is not None and d.crs.to_epsg() != 4326:
    d = d.to_crs(4326)
d["geometry"] = d.geometry.simplify(a.tol, preserve_topology=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
if os.path.exists(a.out):
    os.remove(a.out)
d.to_file(a.out, driver="GeoJSON", COORDINATE_PRECISION=4)

mb = os.path.getsize(a.out) / 1e6
print(f"Wrote {len(d)} boundaries to {a.out} ({mb:.1f} MB)")
if mb > 10:
    print("Over 10 MB: rerun with a larger --tol, e.g. --tol 0.006")
