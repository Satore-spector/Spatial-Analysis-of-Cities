"""Step 5a: classical, rule-based measures of urban structure for each city.

These do not use the neural network at all. They are the yardstick the
network's groups are compared against in step 5b (analysis.compare).

Activity measure: POIs per km2 by default (a stand-in for employment, which
the classical literature uses but OpenStreetMap does not have). Change it with
--measure, e.g. --measure poi_offices or --measure buildings.

The city grid is first merged into --vcell metre cells (default 500 m), and
the CBD is the busiest point found in step 2. Then, for each city:

1. Density gradient (the monocentric model). Fit  log(density) = a - b * distance.
   A steep slope b and a high R2 mean activity falls away smoothly from one
   centre; a low R2 means the pattern is lumpier than one centre explains.

2. Giuliano-Small style centres. Cells above the city's --density-pct
   percentile of density are joined into contiguous groups; a group counts as
   a centre if it holds at least --min-share of all the city's activity. The
   group at the CBD is the main centre; the rest are subcentres.
   (Giuliano and Small (1991) used fixed cut-offs of 10 jobs/acre and 10,000
   jobs; here the cut-offs are relative, because POI counts are not jobs and
   vary with mapping quality.)

3. McMillen style subcentres. Fit a smooth curve of log density against
   distance from the CBD (LOWESS). Cells far above that curve (standardised
   residual above --z) are candidates; contiguous candidate groups holding at
   least --min-share of activity and lying at least --min-dist-km from the CBD
   are subcentres. (A simplified version of McMillen (2001), which uses a
   locally weighted regression over both coordinates.)

A city is called monocentric by a test if that test finds no subcentres.

Writes data/analysis/classical/classical_measures.csv and one map per city
(classical_<slug>.png) with the centres outlined.

Usage:
    python -m analysis.classical
    python -m analysis.classical --measure poi_offices --vcell 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .common import DATA_DIR, block_reduce, find_cities, load_city_grid, raster_dir, say, write_json

EIGHT = np.ones((3, 3), int)  # cells touching at an edge or a corner count as joined


def activity_layer(grid, meta, measure):
    ch = meta["channels"]
    if measure == "pois":
        return grid[[i for i, c in enumerate(ch) if c.startswith("poi_")]].sum(0)
    if measure not in ch:
        raise SystemExit(f"Unknown --measure '{measure}'. Use 'pois' or one of: {', '.join(ch)}")
    return grid[ch.index(measure)]


def centres_from_candidates(cand, act, cbd, dist_km, min_share, min_dist_km, total):
    """Join candidate cells into groups and keep those big enough.

    Returns (labelled array, list of dicts for the kept groups). The group
    containing the CBD (or nearest to it) is marked main=True.
    """
    lab, n = ndimage.label(cand, structure=EIGHT)
    groups = []
    for g in range(1, n + 1):
        cells = lab == g
        share = act[cells].sum() / total if total > 0 else 0
        if share < min_share:
            lab[cells] = 0
            continue
        groups.append({"id": g, "share": share, "min_dist_km": float(dist_km[cells].min()),
                       "contains_cbd": bool(cells[cbd])})
    if groups:
        main = next((g for g in groups if g["contains_cbd"]), None) or min(groups, key=lambda g: g["min_dist_km"])
        for g in groups:
            g["main"] = g is main and g["min_dist_km"] < min_dist_km
    keep = [g for g in groups if g["main"] or g["min_dist_km"] >= min_dist_km]
    for g in groups:
        if g not in keep:
            lab[lab == g["id"]] = 0
    return lab, keep


def lowess_curve(dist, logd, frac):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    fit = lowess(logd, dist, frac=frac, return_sorted=True)
    return np.interp(dist, fit[:, 0], fit[:, 1])


def measure_city(slug, data_dir, measure, vcell, density_pct, min_share, z_crit, min_dist_km, frac):
    grid, mask, meta = load_city_grid(raster_dir(data_dir) / f"{slug}.npz")
    cell = meta["cell_m"]
    f = max(1, int(round(vcell / cell)))
    act = block_reduce(activity_layer(grid, meta, measure), f, "sum")
    inside = block_reduce(mask.astype(float), f, "mean") >= 0.5
    act[~inside] = 0
    km2 = (cell * f / 1000) ** 2
    dens = act / km2
    cbd = (min(meta["centre_rc"][0] // f, act.shape[0] - 1), min(meta["centre_rc"][1] // f, act.shape[1] - 1))
    rr, cc = np.indices(act.shape)
    dist_km = np.hypot(rr - cbd[0], cc - cbd[1]) * cell * f / 1000
    total = act.sum()
    pos = inside & (dens > 0)
    row = {"city": slug, "measure": measure, "cell_km": cell * f / 1000, "total_activity": float(total),
           "cells_with_activity": int(pos.sum()), "cells_inside": int(inside.sum())}
    if pos.sum() < 10:
        row.update(status="too_little_data")
        return row, None

    # 1. Density gradient
    slope, intercept = np.polyfit(dist_km[pos], np.log(dens[pos]), 1)
    pred = intercept + slope * dist_km[pos]
    resid = np.log(dens[pos]) - pred
    r2 = 1 - resid.var() / np.log(dens[pos]).var()
    within5 = act[inside & (dist_km <= 5)].sum() / total
    row.update(gradient_per_km=round(-slope, 4), gradient_r2=round(r2, 3),
               share_within_5km=round(within5, 3))

    # 2. Giuliano-Small style
    cut = np.percentile(dens[pos], density_pct)
    gs_lab, gs = centres_from_candidates(inside & (dens >= cut), act, cbd, dist_km,
                                         min_share, min_dist_km, total)
    gs_main = [g for g in gs if g["main"]]
    row.update(gs_density_cutoff=round(cut, 2), gs_centres=len(gs),
               gs_subcentres=sum(not g["main"] for g in gs),
               gs_main_share=round(gs_main[0]["share"], 3) if gs_main else 0.0)

    # 3. McMillen style
    fitted = np.full(dens.shape, np.nan)
    fitted[pos] = lowess_curve(dist_km[pos], np.log(dens[pos]), frac)
    r = np.where(pos, np.log(np.where(pos, dens, 1)) - fitted, np.nan)
    zres = (r - np.nanmean(r)) / np.nanstd(r)
    mc_lab, mc = centres_from_candidates(pos & (np.nan_to_num(zres) >= z_crit), act, cbd, dist_km,
                                         min_share, min_dist_km, total)
    subs = [g for g in mc if not g["main"]]
    row.update(mcmillen_subcentres=len(subs),
               mcmillen_subcentre_share=round(sum(g["share"] for g in subs), 3))

    row["gs_type"] = "monocentric" if row["gs_subcentres"] == 0 else "polycentric"
    row["mcmillen_type"] = "monocentric" if row["mcmillen_subcentres"] == 0 else "polycentric"
    row["classical_type"] = row["gs_type"] if row["gs_type"] == row["mcmillen_type"] else "unclear"
    row["status"] = "ok"
    return row, (dens, inside, gs_lab, mc_lab, cbd)


def city_map(path, slug, dens, inside, gs_lab, mc_lab, cbd, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.log1p(np.where(inside, dens, np.nan)), cmap="Greys", interpolation="nearest")
    if gs_lab.any():
        ax.contour(gs_lab > 0, levels=[0.5], colors="crimson", linewidths=1.2)
    if mc_lab.any():
        ax.contour(mc_lab > 0, levels=[0.5], colors="dodgerblue", linewidths=1.2, linestyles="dashed")
    ax.plot(cbd[1], cbd[0], "+", color="orange", markersize=14, mew=2)
    ax.set_title(f"{slug}: red = Giuliano-Small centres ({row['gs_subcentres']} sub), "
                 f"blue dashed = McMillen subcentres ({row['mcmillen_subcentres']}), + = CBD",
                 fontsize=8)
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def run(data_dir, cities=None, measure="pois", vcell=500, density_pct=90, min_share=0.03,
        z_crit=1.96, min_dist_km=3.0, frac=0.3):
    data_dir = Path(data_dir)
    out = data_dir / "analysis" / "classical"
    out.mkdir(parents=True, exist_ok=True)
    slugs = [s for s in find_cities(data_dir, cities) if (raster_dir(data_dir) / f"{s}.npz").exists()]
    if not slugs:
        raise SystemExit("No city grids found. Run step 2 first: python -m analysis.rasterize")
    rows = []
    for slug in slugs:
        try:
            row, maps = measure_city(slug, data_dir, measure, vcell, density_pct, min_share,
                                     z_crit, min_dist_km, frac)
        except Exception as e:
            row, maps = {"city": slug, "status": "error", "message": str(e)}, None
        rows.append(row)
        if maps is not None:
            city_map(out / f"classical_{slug}.png", slug, *maps, row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "classical_measures.csv", index=False)
    write_json(out / "classical_settings.json", {
        "measure": measure, "vcell_m": vcell, "density_pct": density_pct, "min_share": min_share,
        "z": z_crit, "min_dist_km": min_dist_km, "lowess_frac": frac})
    cols = [c for c in ["city", "gradient_per_km", "gradient_r2", "gs_subcentres",
                        "mcmillen_subcentres", "classical_type", "status"] if c in res]
    say("Classical measures\n" + res[cols].to_string(index=False))
    return res


def add_args(ap):
    ap.add_argument("--measure", default="pois", help="'pois' (all POIs) or a layer name")
    ap.add_argument("--vcell", type=float, default=500, help="cell size in metres for these tests")
    ap.add_argument("--density-pct", type=float, default=90,
                    help="Giuliano-Small: density percentile a cell must reach")
    ap.add_argument("--min-share", type=float, default=0.03,
                    help="a centre must hold at least this share of the city's activity")
    ap.add_argument("--z", type=float, default=1.96, help="McMillen: residual cut-off")
    ap.add_argument("--min-dist-km", type=float, default=3.0,
                    help="a subcentre must be at least this far from the CBD")
    ap.add_argument("--lowess-frac", type=float, default=0.3, help="McMillen: smoothness of the curve")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    add_args(ap)
    a = ap.parse_args(argv)
    run(a.data_dir, a.cities, a.measure, a.vcell, a.density_pct, a.min_share, a.z,
        a.min_dist_km, a.lowess_frac)


if __name__ == "__main__":
    main()
