"""Final step for a country run: one type per city, with the evidence behind it.

GHSL comes first. The classical centre tests (density gradient, Giuliano-Small
style centres, McMillen style subcentres; see classical.py) are run on the
satellite built-up layer, which covers every city equally. That gives each
city's main type. The same tests on OpenStreetMap POIs are used as support,
and only where the city is reasonably well mapped. The neural network's
groups are reported next to the types, as a check, not used to set them.

Rules (written out in the report too):
  - GHSL tests agree (both find no subcentres, or both find some):
        that type. Confidence "high" if the OSM tests agree and the city is not
        sparsely mapped, otherwise "medium".
  - GHSL tests disagree: the OSM type if the city is not very sparsely mapped
        and the OSM tests agree with each other (confidence "low"); otherwise "unclear".
  - No GHSL data: the OSM type with confidence "low", or "no data".

OSM mapping quality: POIs per km2 of built-up area (from GHSL), compared with
the median city. "good" at half the median or more, "sparse" from 15% to
half, "very sparse" below 15%. If the network's groups follow this ratio, they
are partly sorting cities by mapping effort; the report tests for that.

Reads (all under --data-dir):
  uc/<cc>_cities.csv                         the city list (population, area)
  rasters/<slug>.npz, rasters/rasterize_summary.csv
  analysis/classical/classical_measures_built_up.csv   GHSL tests
  analysis/classical/classical_measures.csv            OSM POI tests
  analysis/tile|city/city_clusters.csv (optional)       network groups
Writes, to <data-dir>/analysis/classification/:
  city_types.csv     one row per city: type, confidence, evidence
  city_types.gpkg    the same with each city's boundary, for QGIS
  city_types_map.png every city on a map, by type and by network group
  report.md          the answer in plain words

Usage:
    python -m analysis.classify --data-dir data/india --cities-csv data/india/uc/india_cities.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .classical import measures_file
from .common import analysis_dir, load_city_grid, md_table, raster_dir, say

TYPES = ["monocentric", "polycentric"]
SIZE_BINS = [0, 5e5, 1e6, 5e6, np.inf]
SIZE_LABELS = ["0.1-0.5 million", "0.5-1 million", "1-5 million", "5 million+"]


def built_up_km2(data_dir: Path, slugs) -> pd.Series:
    """Built-up area inside each city from the GHSL layer of its grid (NaN without GHSL)."""
    out = {}
    for s in slugs:
        f = raster_dir(data_dir) / f"{s}.npz"
        if not f.exists():
            continue
        g, mask, meta = load_city_grid(f)
        if not meta.get("has_ghsl"):
            out[s] = np.nan
            continue
        share = g[meta["channels"].index("built_up")]
        out[s] = float(share[mask].sum()) * (meta["cell_m"] / 1000) ** 2
    return pd.Series(out, name="built_up_km2")


def osm_quality(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unknown"
    return "good" if ratio >= 0.5 else "sparse" if ratio >= 0.15 else "very sparse"


def decide(r) -> tuple[str, str, str]:
    """(final_type, confidence, reason) for one city."""
    g, o, q = r.get("ghsl_type"), r.get("osm_type"), r.get("osm_quality")
    osm_usable = q in ("good", "sparse")
    if g in TYPES:
        if o == g and q == "good":
            return g, "high", "GHSL tests agree; OSM tests agree"
        why = ("OSM tests disagree" if o in TYPES else "OSM tests unclear") if q == "good" else f"OSM {q}"
        return g, "medium", f"GHSL tests agree; {why}"
    if g == "unclear":
        if osm_usable and o in TYPES:
            return o, "low", "GHSL tests disagree; type from OSM tests"
        return "unclear", "", "GHSL tests disagree; OSM cannot settle it"
    if osm_usable and o in TYPES:
        return o, "low", "no GHSL result; type from OSM tests"
    return "no data", "", "no usable GHSL or OSM result"


def load_classical(data_dir, measure, prefix) -> pd.DataFrame | None:
    f = measures_file(data_dir, measure)
    if not f.exists():
        return None
    d = pd.read_csv(f)
    keep = ["city", "classical_type", "gs_subcentres", "mcmillen_subcentres", "gradient_r2",
            "gradient_per_km", "share_within_5km", "status"]
    d = d[[c for c in keep if c in d]].rename(columns={"classical_type": "type"})
    d.loc[d.get("status", "ok") != "ok", "type"] = np.nan
    d = d.drop(columns=[c for c in ["status"] if c in d])
    return d.rename(columns={c: f"{prefix}_{c}" for c in d.columns if c != "city"})


def network_columns(data_dir, slugs) -> pd.DataFrame:
    from .compare import learned_score
    out = pd.DataFrame(index=pd.Index(slugs, name="city"))
    for mode in ("tile", "city"):
        d = analysis_dir(data_dir, mode)
        f = d / "city_clusters.csv"
        if f.exists():
            cc = pd.read_csv(f).set_index("city")["city_cluster"]
            out[f"network_group_{mode}"] = cc.reindex(out.index)
            sc = learned_score(d, mode)
            if sc is not None:
                out[f"network_score_{mode}"] = sc.reindex(out.index).round(3)
    return out.reset_index()


def run(data_dir, cities_csv=None):
    data_dir = Path(data_dir)
    out = data_dir / "analysis" / "classification"
    out.mkdir(parents=True, exist_ok=True)

    if cities_csv is None:
        found = sorted((data_dir / "uc").glob("*_cities.csv"))
        cities_csv = found[0] if found else None
    lst = pd.read_csv(cities_csv) if cities_csv else None
    rs = raster_dir(data_dir) / "rasterize_summary.csv"
    if not rs.exists():
        raise SystemExit("No rasterize_summary.csv; run step 2 first.")
    ras = pd.read_csv(rs)
    ras = ras[ras["status"] == "ok"]
    df = pd.DataFrame({"city": ras["city"]})
    if lst is not None:
        info = lst.rename(columns={"slug": "city", "city": "label"})
        keep = [c for c in ["city", "label", "name", "state", "pop", "area_km2", "large_area", "lon", "lat"]
                if c in info]
        df = df.merge(info[keep], on="city", how="left")
    df = df.merge(ras[["city", "pois", "centre_lon", "centre_lat"]], on="city", how="left")
    df["built_up_km2"] = df["city"].map(built_up_km2(data_dir, df["city"])).round(2)

    ghsl = load_classical(data_dir, "built_up", "ghsl")
    osm = load_classical(data_dir, "pois", "osm")
    if ghsl is None:
        say("  No GHSL classical results (classical_measures_built_up.csv); types will rest on OSM only.")
    for d in (ghsl, osm):
        if d is not None:
            df = df.merge(d, on="city", how="left")
    for col in ("ghsl_type", "osm_type"):
        if col not in df:
            df[col] = np.nan

    dens = df["pois"] / df["built_up_km2"]
    if dens.notna().sum() == 0 and "area_km2" in df:   # no GHSL at all: POIs per km2 of city
        dens = df["pois"] / df["area_km2"]
    df["pois_per_built_km2"] = dens.round(1)
    df["osm_ratio_to_median"] = (dens / dens.median()).round(3)
    df["osm_quality"] = df["osm_ratio_to_median"].map(osm_quality)

    decided = df.apply(lambda r: decide(r), axis=1, result_type="expand")
    df["final_type"], df["confidence"], df["reason"] = decided[0], decided[1], decided[2]
    if "pop" in df:
        df["size_class"] = pd.cut(df["pop"], SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    df = df.merge(network_columns(data_dir, df["city"]), on="city", how="left")
    df = df.sort_values("pop" if "pop" in df else "city", ascending="pop" not in df).reset_index(drop=True)
    df.to_csv(out / "city_types.csv", index=False)
    write_gpkg(df, data_dir, out)
    draw_map(df, out)
    report = make_report(df, out)
    say(report.split("## Network groups")[0])
    say(f"Results -> {out}")
    return df


def write_gpkg(df, data_dir, out):
    import geopandas as gpd
    geoms = []
    for s in df["city"]:
        f = data_dir / f"{s}_boundary.gpkg"
        geoms.append(gpd.read_file(f).to_crs(4326).union_all() if f.exists() else None)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs=4326)
    for c in gdf.columns:
        if gdf[c].dtype == bool:
            gdf[c] = gdf[c].astype(int)
    gdf.to_file(out / "city_types.gpkg", driver="GPKG")


def draw_map(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = df["lon"] if "lon" in df else df["centre_lon"]
    lat = df["lat"] if "lat" in df else df["centre_lat"]
    size = 8 + 120 * np.sqrt(df["pop"] / df["pop"].max()) if "pop" in df else 30
    colours = {"monocentric": "#2E86AB", "polycentric": "#E4572E", "unclear": "#B0B0B0", "no data": "#E0E0E0"}
    panels = [("final_type", "Type (GHSL first, OSM as support)")]
    if "network_group_tile" in df and df["network_group_tile"].notna().any():
        panels.append(("network_group_tile", "Network group (tile mode)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 8), squeeze=False)
    for ax, (col, title) in zip(axes[0], panels):
        if col == "final_type":
            for t, c in colours.items():
                sel = df[col] == t
                for conf, face in (("high", True), ("medium", True), ("low", False), ("", True)):
                    s2 = sel & (df["confidence"] == conf)
                    if s2.any():
                        ax.scatter(lon[s2], lat[s2], s=np.asarray(size)[s2.to_numpy()] if np.ndim(size) else size,
                                   c=c if face else "none", edgecolors=c, linewidths=1.2,
                                   alpha=0.85 if conf != "medium" else 0.55)
                ax.scatter([], [], c=c, label=f"{t} ({int(sel.sum())})")
            ax.scatter([], [], c="none", edgecolors="grey", label="hollow = low confidence")
        else:
            cmap = plt.get_cmap("tab10")
            for g in sorted(df[col].dropna().unique()):
                sel = df[col] == g
                ax.scatter(lon[sel], lat[sel], s=np.asarray(size)[sel.to_numpy()] if np.ndim(size) else size,
                           color=cmap(int(g) % 10), alpha=0.8, label=f"group {int(g)} ({int(sel.sum())})")
        if "pop" in df:
            for _, r in df.nlargest(12, "pop").iterrows():
                ax.annotate(str(r.get("name", r["city"])), (r["lon"] if "lon" in df else r["centre_lon"],
                                                            r["lat"] if "lat" in df else r["centre_lat"]),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Every city, sized by population", fontsize=10)
    fig.savefig(out / "city_types_map.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def make_report(df, out) -> str:
    from scipy.stats import kruskal
    from sklearn.metrics import adjusted_rand_score

    n = len(df)
    L = ["# City types", "",
         f"{n} cities. Type from the classical centre tests on GHSL built-up (satellite), with "
         "OpenStreetMap POIs as support. Rules: see analysis/classify.py.", "",
         "## How many of each", "",
         md_table(df["final_type"].value_counts().rename("cities").to_frame()), "",
         md_table(pd.crosstab(df["final_type"], df["confidence"].replace("", "-"))), ""]
    if "size_class" in df:
        L += ["## By city size", "",
              md_table(pd.crosstab(df["size_class"], df["final_type"])), "",
              "Larger cities are expected to be polycentric more often; a strong pattern the other "
              "way would be worth checking.", ""]
    agree = df[df["ghsl_type"].isin(TYPES) & df["osm_type"].isin(TYPES)]
    if len(agree):
        L += ["## GHSL vs OSM tests", "",
              f"Where both give a clear type ({len(agree)} cities) they agree for "
              f"{(agree['ghsl_type'] == agree['osm_type']).mean():.0%}.", "",
              md_table(pd.crosstab(agree["ghsl_type"], agree["osm_type"])), ""]
    L += ["## OpenStreetMap mapping quality", "",
          md_table(df["osm_quality"].value_counts().rename("cities").to_frame()), "",
          "POIs per km2 of built-up area, against the median city. Sparse cities lean on GHSL alone.", ""]

    for mode in ("tile", "city"):
        col = f"network_group_{mode}"
        if col not in df or df[col].isna().all():
            continue
        d = df[df[col].notna()]
        L += [f"## Network groups ({mode} mode) vs types", "",
              md_table(pd.crosstab(d[col].astype(int).rename("network group"), d["final_type"])), ""]
        clear = d[d["final_type"].isin(TYPES)]
        if len(clear) >= 5 and clear[col].nunique() > 1:
            L += [f"- Adjusted Rand index with the types: {adjusted_rand_score(clear['final_type'], clear[col]):.2f} "
                  "(0 = chance, 1 = identical)"]
        sc = f"network_score_{mode}"
        if sc in d and d[sc].notna().sum() >= 5:
            from scipy.stats import spearmanr
            v = d[[sc, "ghsl_gs_subcentres"]].dropna() if "ghsl_gs_subcentres" in d else pd.DataFrame()
            if len(v) >= 5 and v.iloc[:, 1].nunique() > 1:
                r, p = spearmanr(v[sc], v.iloc[:, 1])
                L += [f"- Rank correlation of the network's main score with the number of GHSL subcentres: "
                      f"{abs(r):.2f} (p = {p:.3g})"]
        # Mapping-quality check: do the groups follow how well cities are mapped in OSM?
        L += ["", "Mapping-quality check (network group vs OSM mapping quality):", "",
              md_table(pd.crosstab(d[col].astype(int).rename("network group"), d["osm_quality"])), ""]
        sparse = d["osm_quality"].isin(["sparse", "very sparse"])
        overall = sparse.mean()
        flagged = [int(g) for g, sub in d.groupby(col)
                   if len(sub) >= 2 and sparse[sub.index].mean() >= max(0.5, 2 * overall)]
        q = d[["osm_ratio_to_median", col]].dropna()
        groups = [g["osm_ratio_to_median"].to_numpy() for _, g in q.groupby(col) if len(g) >= 3]
        p = kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
        if flagged or (np.isfinite(p) and p < 0.01):
            L += ["- **Warning:** the network groups follow OSM mapping quality"
                  + (f" (group(s) {', '.join(map(str, flagged))} are mostly sparsely mapped cities)" if flagged else "")
                  + (f"; Kruskal-Wallis p = {p:.3g}" if np.isfinite(p) else "")
                  + ". These groups partly sort cities by mapping effort, not by shape."]
        else:
            L += ["- No sign that the groups follow OSM mapping quality"
                  + (f" (Kruskal-Wallis p = {p:.3g})." if np.isfinite(p) else ".")]
        L += [""]

    poly = df[df["final_type"] == "polycentric"]
    if len(poly) and "pop" in df:
        cols = [c for c in ["label", "pop", "ghsl_gs_subcentres", "ghsl_mcmillen_subcentres", "confidence"] if c in poly]
        L += ["## Largest polycentric cities", "", md_table(poly.nlargest(15, "pop")[cols].set_index(cols[0])), ""]
    if "large_area" in df and df["large_area"].fillna(False).astype(bool).any():
        big = df[df["large_area"].fillna(False).astype(bool)]
        cols = [c for c in ["label", "pop", "area_km2", "final_type"] if c in big]
        L += ["## Very large urban centres (check by hand)", "",
              "The GHSL database joins built-up areas that touch, so some of these are several towns "
              "counted as one city. Their type says more about the region than about one city.", "",
              md_table(big[cols].set_index(cols[0])), ""]
    L += ["## Caveats", "",
          "- The classical tests here count built-up area (GHSL) or POIs, not jobs, and use relative "
          "cut-offs (see analysis/classical.py).",
          "- GHSL built-up is from satellite imagery for one year (the epoch downloaded); OSM is today's map.",
          "- Network groups are a check: they are learned without labels and need not line up with the "
          "two classical types.", ""]
    text = "\n".join(L)
    (out / "report.md").write_text(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cities-csv", default=None, help="default: the *_cities.csv in <data-dir>/uc/")
    a = ap.parse_args(argv)
    run(a.data_dir, a.cities_csv)


if __name__ == "__main__":
    main()
