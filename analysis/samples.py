"""Step 3a: cut the city grids into the training examples the network sees.

Two modes (the --mode switch):

  city   one example per city: a fixed-size square window centred on the
         city's busiest point, merged into --city-px x --city-px pixels.
         Matches the monocentric/polycentric question directly, but you need
         dozens to hundreds of cities before a network can learn much.

  tile   many examples per city: --tile-px x --tile-px squares of the base
         grid (3.2 km at the default 100 m cells and 32 px). Even a handful of
         cities gives thousands of examples. A city is then described by its
         mix of tile types (see cluster.py).

Normalisation (--normalise):

  city-share (default)  each layer is divided by that city's own average
         before taking logs, so the network sees "how much busier than this
         city's average is this cell", not raw counts. This stops it sorting
         cities by how thoroughly they are mapped in OpenStreetMap (the main
         risk named in the README).
  absolute  raw counts (log scale). Use this to compare levels across cities.

Writes data/analysis/<mode>/samples.npz and samples.csv (one row per example).

Usage:
    python -m analysis.samples --mode tile
    python -m analysis.samples --mode city --city-window-km 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (CHANNEL_AGG, DATA_DIR, analysis_dir, block_reduce, find_cities,
                     load_city_grid, raster_dir, say, write_json)


def choose_channels(grids: dict, requested: str) -> list[str]:
    """Which layers to feed the network.

    'auto' keeps every layer that has data, but drops GHSL built-up unless
    every city has it (a layer that is real in some cities and all zeros in
    others would teach the network which cities had a GHSL file).
    """
    all_ch = next(iter(grids.values()))[2]["channels"]
    if requested != "auto":
        chosen = [c.strip() for c in requested.split(",") if c.strip()]
        bad = [c for c in chosen if c not in all_ch]
        if bad:
            raise SystemExit(f"Unknown channel(s) {bad}. Choose from: {', '.join(all_ch)}")
        return chosen
    chosen = []
    for i, c in enumerate(all_ch):
        if c == "built_up" and not all(m["has_ghsl"] for _, _, m in grids.values()):
            say("  dropping 'built_up': not every city has a GHSL file")
            continue
        if all(g[i].sum() == 0 for g, _, _ in grids.values()):
            say(f"  dropping '{c}': empty in every city")
            continue
        chosen.append(c)
    return chosen


def _window(a: np.ndarray, r0: int, c0: int, size: int) -> np.ndarray:
    """size x size window of a 2-D array starting at (r0, c0); zero outside."""
    out = np.zeros((size, size), dtype=a.dtype)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + size, a.shape[0]), min(c0 + size, a.shape[1])
    if rs < re and cs < ce:
        out[rs - r0:re - r0, cs - c0:ce - c0] = a[rs:re, cs:ce]
    return out


def city_samples(grids, channels, city_px, window_km):
    xs, rows = [], []
    for slug, (grid, mask, meta) in grids.items():
        cell = meta["cell_m"]
        factor = max(1, round(window_km * 1000 / cell / city_px))
        win = city_px * factor
        cr, cc = meta["centre_rc"]
        r0, c0 = cr - win // 2, cc - win // 2
        idx = [meta["channels"].index(c) for c in channels]
        layers = [block_reduce(_window(grid[i], r0, c0, win), factor, CHANNEL_AGG[c])
                  for i, c in zip(idx, channels)]
        m = block_reduce(_window(mask.astype(float), r0, c0, win), factor, "mean")
        xs.append(np.stack(layers))
        rows.append({"sample_id": slug, "city": slug, "row0": r0, "col0": c0, "size_cells": win,
                     "pixel_m": cell * factor, "inside_share": round(float(m.mean()), 3),
                     "dist_km": 0.0})
        city_cover = mask[max(r0, 0):r0 + win, max(c0, 0):c0 + win].sum() / max(mask.sum(), 1)
        if city_cover < 0.9:
            say(f"  {slug}: window covers only {100 * city_cover:.0f}% of the city; "
                f"consider a larger --city-window-km")
    say(f"  city window: {city_px} px of {rows[0]['pixel_m']:g} m = "
        f"{city_px * rows[0]['pixel_m'] / 1000:g} km across")
    return xs, rows


def tile_samples(grids, channels, tile_px, stride, min_inside, min_activity, activity_layers="all"):
    xs, rows = [], []
    for slug, (grid, mask, meta) in grids.items():
        cell = meta["cell_m"]
        cr, cc = meta["centre_rc"]
        idx = [meta["channels"].index(c) for c in channels]
        sub = grid[idx]
        if activity_layers == "poi":   # POI counts only, so --min-activity means "POIs per tile"
            activity = grid[[i for i, c in enumerate(meta["channels"]) if c.startswith("poi_")]].sum(0)
        else:                          # every layer; mixes counts, shares and road metres
            activity = grid.sum(0)
        n0 = len(rows)
        for r0 in range(0, grid.shape[1] - tile_px + 1, stride):
            for c0 in range(0, grid.shape[2] - tile_px + 1, stride):
                inside = mask[r0:r0 + tile_px, c0:c0 + tile_px].mean()
                if inside < min_inside:
                    continue
                if activity[r0:r0 + tile_px, c0:c0 + tile_px].sum() < min_activity:
                    continue
                xs.append(sub[:, r0:r0 + tile_px, c0:c0 + tile_px])
                dist = np.hypot(r0 + tile_px / 2 - cr, c0 + tile_px / 2 - cc) * cell / 1000
                rows.append({"sample_id": f"{slug}_{r0}_{c0}", "city": slug, "row0": r0, "col0": c0,
                             "size_cells": tile_px, "pixel_m": cell,
                             "inside_share": round(float(inside), 3), "dist_km": round(dist, 2)})
        say(f"  {slug}: {len(rows) - n0} tiles")
    say(f"  tile size: {tile_px} px of {cell:g} m = {tile_px * cell / 1000:g} km")
    return xs, rows


def normalise(xs, rows, grids, channels, how):
    """Log-scale, optionally relative to each city's average, then put every
    layer on a similar 0-1-ish range (divide by its 99th percentile)."""
    x = np.stack(xs).astype(np.float32)
    if how == "city-share":
        for slug, (grid, mask, meta) in grids.items():
            sel = np.array([r["city"] == slug for r in rows])
            for k, c in enumerate(channels):
                layer = grid[meta["channels"].index(c)]
                # Average per base cell inside the city, rescaled to the sample's pixel size.
                pix = rows[np.argmax(sel)]["pixel_m"] / meta["cell_m"]
                per_pix = layer[mask].mean() * (pix ** 2 if CHANNEL_AGG[c] == "sum" else 1)
                x[sel, k] = x[sel, k] / per_pix if per_pix > 0 else 0
    x = np.log1p(np.clip(x, 0, None))
    scale = np.ones(len(channels), dtype=np.float32)
    for k in range(len(channels)):
        v = x[:, k][x[:, k] > 0]
        if v.size:
            scale[k] = np.percentile(v, 99)
    x = np.clip(x / scale[None, :, None, None], 0, 3)
    return x, scale


def smooth(grid, mask, meta, smooth_m):
    """Blur every layer slightly (Gaussian, sigma = smooth_m metres).

    At 100 m a single shop is a lone bright pixel, and the network would spend
    its effort on that speckle. A light blur turns points into local density,
    which is what the question is about. Totals are kept; nothing leaks
    outside the boundary.
    """
    from scipy import ndimage
    sigma = smooth_m / meta["cell_m"]
    out = np.stack([ndimage.gaussian_filter(layer.astype(np.float64), sigma, mode="constant")
                    for layer in grid]).astype(np.float32)
    out[:, ~mask] = 0
    return out, mask, meta


def build(data_dir, mode, cities=None, city_px=64, city_window_km=40, tile_px=32,
          tile_stride=None, min_inside=0.5, min_activity=1.0, how="city-share", channels="auto",
          smooth_m=200.0, activity_layers="all"):
    data_dir = Path(data_dir)
    slugs = [s for s in find_cities(data_dir, cities) if (raster_dir(data_dir) / f"{s}.npz").exists()]
    if not slugs:
        raise SystemExit("No city grids found. Run step 2 first: python -m analysis.rasterize")
    for n in (city_px, tile_px):
        if n & (n - 1) or n < 8:
            raise SystemExit("--city-px and --tile-px must be powers of two (16, 32, 64, 128 ...)")
    grids = {s: load_city_grid(raster_dir(data_dir) / f"{s}.npz") for s in slugs}
    cells = {m["cell_m"] for _, _, m in grids.values()}
    if len(cells) > 1:
        raise SystemExit(f"City grids use different cell sizes {cells}; redo step 2 with one --cell.")
    say(f"Building {mode} samples from {len(slugs)} cities")
    chosen = choose_channels(grids, channels)
    if smooth_m > 0:
        grids = {s: smooth(*g, smooth_m) for s, g in grids.items()}
    if mode == "city":
        xs, rows = city_samples(grids, chosen, city_px, city_window_km)
    else:
        xs, rows = tile_samples(grids, chosen, tile_px, tile_stride or tile_px, min_inside, min_activity,
                                activity_layers)
    if not xs:
        raise SystemExit("No samples were made; try a lower --min-inside or --min-activity.")
    x, scale = normalise(xs, rows, grids, chosen, how)

    out = analysis_dir(data_dir, mode)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", x=x, channels=np.array(chosen))
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "samples.csv", index=False)
    write_json(out / "samples_settings.json", {
        "mode": mode, "cities": slugs, "channels": chosen, "normalise": how,
        "scale_99pct": scale.tolist(), "city_px": city_px, "city_window_km": city_window_km,
        "tile_px": tile_px, "tile_stride": tile_stride or tile_px, "smooth_m": smooth_m, "min_inside": min_inside,
        "min_activity": min_activity, "n_samples": int(x.shape[0])})
    say(f"  {x.shape[0]} samples of shape {tuple(x.shape[1:])} -> {out / 'samples.npz'}")
    if mode == "city" and x.shape[0] < 30:
        say(f"  NOTE: only {x.shape[0]} cities. A network cannot learn much from so few; treat "
            "city-mode results as a check that the pipeline runs, or use --mode tile.")
    return x, meta


def add_args(ap):
    ap.add_argument("--city-px", type=int, default=64, help="city mode: image size in pixels")
    ap.add_argument("--city-window-km", type=float, default=40,
                    help="city mode: width of the square window around the centre")
    ap.add_argument("--tile-px", type=int, default=32, help="tile mode: tile size in grid cells")
    ap.add_argument("--tile-stride", type=int, default=None,
                    help="tile mode: step between tiles in cells (default = tile size, no overlap)")
    ap.add_argument("--min-inside", type=float, default=0.5,
                    help="tile mode: keep tiles at least this share inside the boundary")
    ap.add_argument("--min-activity", type=float, default=1.0,
                    help="tile mode: skip tiles with (almost) nothing in them")
    ap.add_argument("--activity", choices=["all", "poi"], default="all",
                    help="tile mode: what --min-activity counts. 'all' sums every layer (counts, "
                         "shares and road metres mixed); 'poi' counts POIs only")
    ap.add_argument("--smooth-m", type=float, default=200,
                    help="blur layers by this many metres before cutting (0 = off)")
    ap.add_argument("--normalise", choices=["city-share", "absolute"], default="city-share")
    ap.add_argument("--channels", default="auto",
                    help="comma-separated layers to use, e.g. poi_offices,poi_retail,buildings")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--mode", choices=["city", "tile"], required=True)
    add_args(ap)
    a = ap.parse_args(argv)
    build(a.data_dir, a.mode, a.cities, a.city_px, a.city_window_km, a.tile_px, a.tile_stride,
          a.min_inside, a.min_activity, a.normalise, a.channels, a.smooth_m, a.activity)


if __name__ == "__main__":
    main()
