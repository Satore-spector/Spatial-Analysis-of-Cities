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
    slug [slug ...]             only these cities

Each file: grid size, cell size in km, the city mask, up to four layers
(built_up, buildings, pois, roads) scaled 0-255 per city, the main centre and
the Giuliano-Small / McMillen subcentres. Numbers are packed as base64 bytes.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

from analysis.classical import measure_city
from analysis.common import CHANNELS, load_city_grid, raster_dir

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


def export_city(slug, data_dir, out_dir, max_dim, measure):
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
    a = p.parse_args()

    rdir = raster_dir(a.data_dir)
    if not rdir.exists():
        sys.exit(f"Not found: {rdir}\nRun from the project root, or pass --data-dir.")
    slugs = a.cities or sorted(f.stem for f in rdir.glob("*.npz"))
    if not slugs:
        sys.exit(f"No .npz city grids in {rdir}. Run step 2 first: python -m analysis.rasterize")

    done, total = 0, 0
    for i, s in enumerate(slugs, 1):
        try:
            size, why = export_city(s, a.data_dir, a.out, a.max_dim, a.measure)
        except Exception as e:
            size, why = None, str(e)
        if size:
            done += 1
            total += size
        else:
            print(f"  {s}: skipped ({why})")
        if i % 100 == 0:
            print(f"  {i}/{len(slugs)}")
    print(f"Wrote {done} city plots to {a.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
