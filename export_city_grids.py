"""Export a small map plot of every city for the website.

The website's results map only knows one point per city (city_types.csv) and a
simplified outline (city_boundaries.geojson). Step 2 of the pipeline saves much
more: a grid per city with POIs, building footprints, roads and GHSL built-up
(data/<run>/rasters/<slug>.npz). This script shrinks those grids and adds the
centres found by the classical tests, so the page can draw a real map of the
city when you click it.

Run from the project root, after the pipeline has finished:
    python export_city_grids.py
Options:
    --data-dir data/india       folder that holds rasters/<slug>.npz
    --out site/public/city_grids  one <slug>.json per city is written here
    --max-dim 56                longest side of a plot in cells (bigger = sharper, larger files)
    --measure auto              activity used to find centres: auto, pois or built_up
    --max-pois 1200             POI dots kept per city (random sample, all categories)
    --max-road-pts 6000         vertices kept for roads per city (major roads first)
    --max-built-pts 1500        vertices kept for built-up outlines per city
    --no-osm                    skip the Map view data (grids only)
    --skip-existing             keep city files that are already in --out (resume an interrupted run)
    slug [slug ...]             only these cities

Each file: grid size, cell size in km, the city mask, up to four layers
(built_up, buildings, pois, roads) scaled 0-255 per city, the main centre and
the Giuliano-Small / McMillen subcentres. Numbers are packed as base64 bytes.

It also writes an "osm" block for the Map view on the website, read straight
from the city's OSM files (data/<run>/<slug>_pois.gpkg, _roads.gpkg, and the
built-up surface from the saved grid): POI points by category, major and minor
roads as simplified lines, and built-up outlines as polygons. All are in the
same coordinates as the grid plot. They are thinned to a size budget so a city
file stays small (see --max-pois, --max-road-pts, --max-built-pts).
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

from analysis.classical import measure_city
from analysis.common import (CHANNELS, MAJOR_ROADS, POI_CATEGORIES, load_city_grid, poi_category,
                             raster_dir)

U = 16          # Map view coordinates: 16 steps per plot cell (a plot is ~56 cells wide)

LAYERS = {
    "built_up": ["built_up"],
    "buildings": ["buildings"],
    "pois": [c for c in CHANNELS if c.startswith("poi_")],
    "roads": ["roads_major", "roads_minor"],
}


def block_sum(a, f):
    """Add up f x f blocks. The array is padded with zeros so no edge is lost."""
    h, w = -(-a.shape[0] // f) * f, -(-a.shape[1] // f) * f
    p = np.zeros((h, w), float)
    p[:a.shape[0], :a.shape[1]] = a
    return p.reshape(h // f, f, w // f, f).sum(axis=(1, 3))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def centre_groups(lab, dens, cbd, scale, r0, c0):
    """Centre point and size of each kept group, in plot-cell coordinates."""
    out = []
    total = float(dens.sum()) or 1.0
    for g in np.unique(lab[lab > 0]):
        cells = lab == g
        rr, cc = np.nonzero(cells)
        out.append({
            "x": round(float((cc.mean() + .5) * scale[1] - c0), 2),
            "y": round(float((rr.mean() + .5) * scale[0] - r0), 2),
            "r": round(float(np.sqrt(cells.sum() / np.pi) * scale[0]), 2),
            "share": round(float(dens[cells].sum() / total), 3),
            "main": bool(lab[cbd] == g),
        })
    return out


def b64i16(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<i2").tobytes()).decode("ascii")


def path_b64(xy):
    """Flat [x0, y0, dx, dy, ...] as base64 int16: first point absolute, then steps."""
    q = np.round(np.asarray(xy) * U).astype(np.int64)
    d = np.vstack([q[:1], np.diff(q, axis=0)])
    return b64i16(d.ravel())


def osm_layers(slug, data_dir, meta, grid, mask, r0, r1, c0, c1, f, inside, n_pois, n_road, n_built):
    """POIs, roads and built-up outlines of one city, in plot-cell coordinates (x right, y down)."""
    import geopandas as gpd
    import shapely
    from pyproj import CRS
    from rasterio import features
    from rasterio.transform import Affine
    from scipy import ndimage

    crs = CRS.from_wkt(meta["crs"])
    t = meta["transform"]
    x0, y1, cell = t[2], t[5], meta["cell_m"]
    h, w = inside.shape
    rng = np.random.default_rng(1)
    out = {"u": U}

    def to_plot(x, y):
        return (x - x0) / (cell * f) - c0 / f, (y1 - y) / (cell * f) - r0 / f

    near = np.pad(ndimage.binary_dilation(inside, iterations=1), 1)   # city plus one cell of slack

    def in_city(px, py):
        """True where a plot-space point falls on a city cell (one cell of slack at the edge)."""
        cc, rr = np.floor(px).astype(int) + 1, np.floor(py).astype(int) + 1
        ok = (cc >= 0) & (cc < w + 2) & (rr >= 0) & (rr < h + 2)
        res = np.zeros(len(px), bool)
        res[ok] = near[rr[ok], cc[ok]]
        return res

    # POIs: one dot each, by category (same rules as the pipeline)
    path = os.path.join(str(data_dir), f"{slug}_pois.gpkg")
    if os.path.exists(path):
        pois = gpd.read_file(path).to_crs(crs)
        pois = pois[pois.geometry.notna() & ~pois.geometry.is_empty]
        cats = poi_category(pois).to_numpy()
        pt = pois.geometry.representative_point()
        px, py = to_plot(pt.x.to_numpy(), pt.y.to_numpy())
        keep = in_city(px, py)
        px, py, cats = px[keep], py[keep], cats[keep]
        if len(px) > n_pois:
            pick = np.sort(rng.choice(len(px), n_pois, replace=False))
            px, py, cats = px[pick], py[pick], cats[pick]
        names, pts = [], []
        for cat in POI_CATEGORIES:
            m = cats == cat
            if m.any():
                names.append(cat)
                pts.append(b64i16(np.round(np.column_stack([px[m], py[m]]) * U).ravel()))
        out["pois"] = {"cats": names, "pts": pts}

    # Roads: simplified lines; major roads first, then the longest minor ones until the budget is used
    path = os.path.join(str(data_dir), f"{slug}_roads.gpkg")
    if os.path.exists(path):
        roads = gpd.read_file(path).to_crs(crs)
        roads = roads.explode(index_parts=False)
        roads = roads[roads.geom_type == "LineString"]
        major = roads["highway"].isin(MAJOR_ROADS).to_numpy() if "highway" in roads else np.zeros(len(roads), bool)
        geoms = roads.geometry.to_numpy()
        mid = shapely.get_coordinates(shapely.line_interpolate_point(geoms, 0.5, normalized=True))
        keep = in_city(*to_plot(mid[:, 0], mid[:, 1]))          # drop roads outside the city
        kept = {"major": [], "minor": []}
        budget = n_road
        for kind, sel in (("major", keep & major), ("minor", keep & ~major)):
            if not sel.any():
                continue
            # OSM cuts streets into many short pieces: join them into whole streets first
            streets = shapely.get_parts(shapely.line_merge(shapely.union_all(geoms[sel])))
            streets = shapely.simplify(streets, cell * f * 0.1)
            length = shapely.length(streets)
            for i in np.argsort(-length):                       # longest streets first
                g = streets[i]
                if g.is_empty or len(g.coords) < 2 or length[i] < cell * f * 0.1:
                    continue
                xy = np.column_stack(to_plot(*np.asarray(g.coords).T))
                if len(xy) > budget:
                    continue
                kept[kind].append(path_b64(xy))
                budget -= len(xy)
        out["roads"] = kept

    # Built-up area: smoothed built-up share (GHSL, else building footprints) cut into outlines
    ch = meta["channels"]
    src = "built_up" if meta.get("has_ghsl") and "built_up" in ch else "buildings"
    f2 = max(1, int(round(f / 3)))
    cr = grid[ch.index(src), r0:r1, c0:c1].astype(float)
    mm = mask[r0:r1, c0:c1].astype(float)
    v = block_sum(cr, f2) / f2 ** 2
    inn = block_sum(mm, f2) >= (f2 * f2) / 2
    v = ndimage.gaussian_filter(np.where(inn, v, 0), 0.8)
    thr = 0.25 if src == "built_up" else 0.06
    on = (v >= thr) & inn
    on = ndimage.binary_closing(on, iterations=1) & inn
    on = ndimage.binary_fill_holes(on) if on.any() else on
    if on.any():
        k = f2 / f                                   # one fine cell, in plot cells
        tol = 0.35 * k
        for _ in range(8):
            rings = []
            for geom, val in features.shapes(on.astype("uint8"), mask=on, transform=Affine.identity()):
                poly = shapely.geometry.shape(geom).simplify(tol * f / f2)
                for p in ([poly] if poly.geom_type == "Polygon" else list(poly.geoms)):
                    if p.is_empty or p.area * k * k < 0.5:   # drop specks under half a plot cell
                        continue
                    for ring in [p.exterior, *p.interiors]:
                        if len(ring.coords) > 3 and shapely.Polygon(ring).area * k * k >= 0.25:
                            rings.append(np.asarray(ring.coords)[:-1] * k)
            if sum(len(r) for r in rings) <= n_built:
                break
            tol *= 1.6
        out["built"] = [path_b64(r) for r in rings]
        out["built_src"] = src
    return out


def export_city(slug, data_dir, out_dir, max_dim, measure, osm=None):
    grid, mask, meta = load_city_grid(raster_dir(data_dir) / f"{slug}.npz")
    ch = meta["channels"]
    cell = meta["cell_m"]
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None, "empty mask"
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1   # crop to the city
    f = max(1, int(np.ceil(max(r1 - r0, c1 - c0) / max_dim)))
    m = mask[r0:r1, c0:c1].astype(float)
    n = block_sum(m, f)
    inside = n >= (f * f) / 2
    out = {"v": 1, "w": int(inside.shape[1]), "h": int(inside.shape[0]),
           "cell_km": round(cell * f / 1000, 3), "mask": b64(np.packbits(inside)), "layers": {}}

    for name, chans in LAYERS.items():
        idx = [ch.index(c) for c in chans if c in ch]
        if not idx:
            continue
        s = block_sum(grid[idx, r0:r1, c0:c1].sum(0), f)
        v = np.where(inside, s / np.maximum(n, 1), 0)        # mean per 100 m cell inside the block
        pos = v[v > 0]
        if not len(pos):
            continue
        top = float(np.percentile(pos, 99)) or float(pos.max())
        q = np.clip(v / top, 0, 1) ** 0.5 * 255              # square root keeps the quiet areas visible
        out["layers"][name] = b64(np.round(q).astype(np.uint8))

    use = measure
    if use == "auto":
        use = "built_up" if "built_up" in out["layers"] else "pois"
    out["measure"] = use
    try:
        row, res = measure_city(slug, data_dir, use, 500, 90, 0.03, 1.96, 3.0, 0.3)
    except Exception as e:                                    # keep the plot, drop the centres
        row, res = None, None
        print(f"  {slug}: centres skipped ({e})")
    if res is not None:
        dens, vin, gs_lab, mc_lab, cbd = res
        fv = max(1, int(round(500 / cell)))                  # classical grid cell, in raster cells
        scale = (fv / f, fv / f)
        ro, co = r0 / f, c0 / f
        out["cbd"] = [round(float((cbd[1] + .5) * scale[1] - co), 2), round(float((cbd[0] + .5) * scale[0] - ro), 2)]
        out["gs"] = centre_groups(gs_lab, dens, cbd, scale, ro, co)
        out["mc"] = centre_groups(mc_lab, dens, cbd, scale, ro, co)

    if osm:
        try:
            o = osm_layers(slug, data_dir, meta, grid, mask, r0, r1, c0, c1, f, inside,
                           osm["pois"], osm["road"], osm["built"])
            if o.get("pois") or o.get("roads") or o.get("built"):
                out["osm"] = o
        except Exception as e:                                # the grid plot still works without it
            print(f"  {slug}: map view skipped ({e})")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slug}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return os.path.getsize(path), ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cities", nargs="*")
    p.add_argument("--data-dir", default="data/india")
    p.add_argument("--out", default="site/public/city_grids")
    p.add_argument("--max-dim", type=int, default=56)
    p.add_argument("--measure", default="auto")
    p.add_argument("--max-pois", type=int, default=1200)
    p.add_argument("--max-road-pts", type=int, default=6000)
    p.add_argument("--max-built-pts", type=int, default=1500)
    p.add_argument("--no-osm", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()

    rdir = raster_dir(a.data_dir)
    if not rdir.exists():
        sys.exit(f"Not found: {rdir}\nRun from the project root, or pass --data-dir.")
    slugs = a.cities or sorted(f.stem for f in rdir.glob("*.npz"))
    if not slugs:
        sys.exit(f"No .npz city grids in {rdir}. Run step 2 first: python -m analysis.rasterize")

    osm = None if a.no_osm else {"pois": a.max_pois, "road": a.max_road_pts, "built": a.max_built_pts}
    import time
    done, total = 0, 0
    for i, s in enumerate(slugs, 1):
        if a.skip_existing and os.path.exists(os.path.join(a.out, f"{s}.json")):
            done += 1
            continue
        t0 = time.time()
        try:
            size, why = export_city(s, a.data_dir, a.out, a.max_dim, a.measure, osm)
        except MemoryError:
            size, why = None, "ran out of memory (try --no-osm for this city)"
        except Exception as e:
            size, why = None, f"{type(e).__name__}: {e}"
        took = time.time() - t0
        if size:
            done += 1
            total += size
            if took > 20 or len(slugs) <= 20:
                print(f"  {s}: {size / 1024:.0f} KB in {took:.0f} s")
        else:
            print(f"  {s}: skipped ({why})")
        if i % 25 == 0:
            print(f"  {i}/{len(slugs)}", flush=True)
    print(f"Wrote {done} city plots to {a.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
