"""Make fake cities in exactly the format data_extraction_osm.R writes.

Half are monocentric (one centre) and half polycentric (3-4 centres), so the
right answer is known and every later step can be tested against it. Nothing
is downloaded.

    python tests/make_fake_cities.py --out /tmp/fake_data --n 12

Writes <slug>_boundary/pois/buildings/roads.gpkg for each city, a GHSL-style
GeoTIFF for some of them, and fake_cities_truth.csv (the answer key).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon, box

# Tags that land in each POI category in amenity_wrap.R: (column, value).
CBD_TAGS = [("office", "company"), ("amenity", "bank"), ("shop", "clothes"),
            ("amenity", "restaurant"), ("amenity", "cafe"), ("office", "it"),
            ("shop", "mall"), ("amenity", "atm"), ("tourism", "hotel")]
SUBURB_TAGS = [("amenity", "place_of_worship"), ("amenity", "school"), ("shop", "convenience"),
               ("leisure", "park"), ("amenity", "clinic"), ("amenity", "parking"),
               ("amenity", "fuel"), ("amenity", "pharmacy"), ("amenity", "toilets")]
POI_COLS = ["amenity", "shop", "office", "public_transport", "railway", "tourism",
            "leisure", "healthcare"]


def local_crs(lon, lat):
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m +datum=WGS84")


def blob_points(rng, centres, weights, n, sd):
    """n points around the centres (a mixture of round normal blobs)."""
    w = np.asarray(weights) / np.sum(weights)
    which = rng.choice(len(centres), size=n, p=w)
    return np.asarray(centres)[which] + rng.normal(0, sd, size=(n, 2))


def make_city(rng, slug, kind, out: Path, lon, lat, with_ghsl):
    crs = local_crs(lon, lat)
    radius = rng.uniform(11000, 15000)
    # A slightly irregular round boundary
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r = radius * (1 + 0.08 * np.sin(3 * ang + rng.uniform(0, 6)) + rng.normal(0, 0.02, ang.size))
    boundary = Polygon(np.c_[r * np.cos(ang), r * np.sin(ang)])

    if kind == "monocentric":
        centres = [rng.normal(0, 1500, 2)]
        weights = [1.0]
    else:
        k = int(rng.integers(3, 5))
        start = rng.uniform(0, 2 * np.pi)
        centres = [rng.normal(0, 800, 2)]  # one centre near the middle
        for j in range(k - 1):
            a = start + 2 * np.pi * j / (k - 1) + rng.normal(0, 0.2)
            d = rng.uniform(6500, 8500)
            centres.append(np.array([d * np.cos(a), d * np.sin(a)]))
        weights = [1.0] + list(rng.uniform(0.6, 0.9, k - 1))
    centres = np.array(centres)

    def inside(xy):
        return np.array([boundary.contains(Point(p)) for p in xy])

    # POIs: busy centres (CBD-type tags) plus a spread-out suburban background.
    n_core = int(rng.uniform(3000, 5000))
    n_bg = int(n_core * 0.8)
    core = blob_points(rng, centres, weights, n_core, rng.uniform(900, 1300))
    # Background density falls off with distance from the main centre (a classic
    # monocentric gradient) in both kinds of city.
    bg = rng.normal(0, radius / 2.2, size=(n_bg * 2, 2)) + centres[0]
    bg = bg[inside(bg)][:n_bg]
    core = core[inside(core)]
    rows = []
    for xy, tags in ((core, CBD_TAGS), (bg, SUBURB_TAGS)):
        for i, (x, y) in enumerate(xy):
            col, val = tags[rng.integers(len(tags))]
            rec = {c: None for c in POI_COLS}
            rec[col] = val
            rec.update(osm_id=str(rng.integers(1e9)), name=None, x=x, y=y,
                       poi_source="node" if rng.random() < 0.8 else "area")
            rows.append(rec)
    pois = pd.DataFrame(rows)
    pois_gdf = gpd.GeoDataFrame(
        pois[["osm_id", "name"] + POI_COLS + ["poi_source"]],
        geometry=gpd.points_from_xy(pois.x, pois.y), crs=crs)

    # Buildings: small squares, denser near the centres
    bxy = np.vstack([blob_points(rng, centres, weights, 6000, 1800),
                     rng.normal(0, radius / 2, size=(6000, 2)) + centres[0]])
    bxy = bxy[inside(bxy)]
    sizes = rng.uniform(8, 25, len(bxy))
    blds = gpd.GeoDataFrame({"osm_id": None, "osm_way_id": [str(i) for i in range(len(bxy))],
                             "name": None, "building": "yes"},
                            geometry=[box(x - s, y - s, x + s, y + s) for (x, y), s in zip(bxy, sizes)],
                            crs=crs)

    # Roads: major radial roads out of each centre, minor grid everywhere.
    lines, kinds = [], []
    for c in centres:
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False) + rng.uniform(0, 1):
            lines.append(LineString([c, c + radius * 1.5 * np.array([np.cos(a), np.sin(a)])]))
            kinds.append("primary")
    for v in np.arange(-radius, radius, 1500):
        lines += [LineString([(v, -radius), (v, radius)]), LineString([(-radius, v), (radius, v)])]
        kinds += ["residential", "residential"]
    roads = gpd.GeoDataFrame({"osm_id": [str(i) for i in range(len(lines))], "name": None,
                              "highway": kinds}, geometry=lines, crs=crs)
    roads = gpd.clip(roads, boundary)

    b = gpd.GeoDataFrame(geometry=[boundary], crs=crs)
    for name, g in (("boundary", b), ("pois", pois_gdf), ("buildings", blds), ("roads", roads)):
        g.to_crs(4326).to_file(out / f"{slug}_{name}.gpkg", driver="GPKG")

    if with_ghsl:
        write_fake_ghsl(out, slug, b, centres, weights, crs)
    return {"city": slug, "true_type": kind, "true_centres": len(centres),
            "lon": lon, "lat": lat}


def write_fake_ghsl(out, slug, boundary, centres, weights, crs):
    """GHSL-like built-up raster (values 0-100) in Mollweide, at 50 m to stay small."""
    import rasterio
    from rasterio.transform import from_origin

    moll = CRS.from_string("ESRI:54009")
    bm = boundary.to_crs(moll)
    minx, miny, maxx, maxy = bm.total_bounds
    res = 50
    w, h = int((maxx - minx) / res) + 1, int((maxy - miny) / res) + 1
    xs = minx + (np.arange(w) + 0.5) * res
    ys = maxy - (np.arange(h) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    cm = gpd.GeoSeries(gpd.points_from_xy(centres[:, 0], centres[:, 1]), crs=crs).to_crs(moll)
    val = np.zeros_like(gx)
    for p, wt in zip(cm, weights):
        val += 80 * wt * np.exp(-((gx - p.x) ** 2 + (gy - p.y) ** 2) / (2 * 2500 ** 2))
    val = np.clip(val + 5, 0, 100).astype("float32")
    (out / "ghsl").mkdir(exist_ok=True)
    with rasterio.open(out / "ghsl" / f"{slug}_ghs_built_s_2018_10m.tif", "w", driver="GTiff",
                       height=h, width=w, count=1, dtype="float32", crs=moll,
                       transform=from_origin(minx, maxy, res, res), nodata=np.nan) as dst:
        dst.write(val, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12, help="number of cities (half of each kind)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ghsl-every", type=int, default=3,
                    help="write a GHSL-style raster for every Nth city (1 = all cities)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    truth = []
    for i in range(args.n):
        kind = "monocentric" if i % 2 == 0 else "polycentric"
        slug = f"fake{chr(ord('a') + i // 26)}{chr(ord('a') + i % 26)}{'mono' if i % 2 == 0 else 'poly'}"
        lon, lat = rng.uniform(72, 88), rng.uniform(10, 28)  # spread over several UTM zones
        truth.append(make_city(rng, slug, kind, out, lon, lat, with_ghsl=(i % args.ghsl_every == 0)))
        print("made", slug, kind, flush=True)
    pd.DataFrame(truth).to_csv(out / "fake_cities_truth.csv", index=False)


if __name__ == "__main__":
    main()
