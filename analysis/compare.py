"""Step 5b: do the network's groups line up with the classical measures?

Reads data/analysis/<mode>/city_clusters.csv (step 4) and
data/analysis/classical/classical_measures.csv (step 5a). Writes, in
data/analysis/<mode>/:

  comparison.csv         one row per city: its group and its classical measures
  comparison_report.md   the answer in plain words, with the numbers behind it
  comparison.png         classical measures for each network group

How agreement is measured:
  - a cross-table of network group vs classical type (mono/poly/unclear);
  - "best-match agreement": label each group with its most common classical
    type, then count how many cities that label gets right (1.0 = perfect);
  - adjusted Rand index (ARI): 0 = no better than chance, 1 = identical groupings;
  - Kruskal-Wallis tests: do the groups differ in density-gradient R2 and in
    number of subcentres? (small p = they differ)

With --truth (a CSV with city,true_type, as the fake-city test writes) the
same scores are also given against the known answer.

Usage:
    python -m analysis.compare --mode tile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score

from .common import DATA_DIR, analysis_dir, md_table, say


def best_match(groups: pd.Series, labels: pd.Series) -> float:
    """Share of cities whose label equals the most common label in their group."""
    ok = 0
    for g in groups.unique():
        lab = labels[groups == g]
        ok += lab.value_counts().iloc[0]
    return ok / len(labels)


def scores(groups, labels):
    keep = labels.notna() & (labels != "unclear")
    g, l = groups[keep], labels[keep]
    if len(l) < 2:
        return {"cities": int(len(l)), "best_match": np.nan, "ari": np.nan}
    return {"cities": int(len(l)), "best_match": round(best_match(g, l), 3),
            "ari": round(adjusted_rand_score(l, g), 3)}


def ranking(df, out, mode):
    """Spearman correlation between each city's main learned score and classical measures.

    The learned description of a city is its embedding (city mode) or its mix
    of tile groups (tile mode, from city_profiles.csv). Its first principal
    component is the score.
    """
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if mode == "tile":
        f = out / "city_profiles.csv"
        if not f.exists():
            return {}
        feats = pd.read_csv(f).set_index("city").filter(like="share_group_")
    else:
        emb = pd.read_csv(out / "embeddings.csv").set_index("city")
        feats = emb.filter(regex=r"^z\d+$")
    feats = feats.reindex(df.city).dropna()
    if len(feats) < 5 or feats.shape[1] < 1:
        return {}
    z = StandardScaler().fit_transform(feats) if feats.shape[1] > 1 else feats.to_numpy()
    score = PCA(1).fit_transform(z)[:, 0] if z.shape[1] > 1 else z[:, 0]
    d = df.set_index("city").loc[feats.index]
    targets = {m: d[m] for m in ("gradient_r2", "gs_subcentres", "mcmillen_subcentres") if m in d}
    if "true_type" in d:
        targets["known answer (fake cities)"] = (d.true_type == "polycentric").astype(float)
    res = {}
    for m, v in targets.items():
        ok = v.notna().to_numpy()
        if ok.sum() >= 5 and np.ptp(v[ok]) > 0:
            r, p = spearmanr(score[ok], v[ok])
            res[m] = (abs(r), p)
    return res


def run(data_dir, mode, truth=None):
    out = analysis_dir(data_dir, mode)
    cfile = out / "city_clusters.csv"
    kfile = Path(data_dir) / "analysis" / "classical" / "classical_measures.csv"
    for f, step in ((cfile, f"python -m analysis.cluster --mode {mode}"),
                    (kfile, "python -m analysis.classical")):
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run: {step}")
    df = pd.read_csv(cfile).merge(pd.read_csv(kfile), on="city", how="left")
    if truth:
        df = df.merge(pd.read_csv(truth)[["city", "true_type"]], on="city", how="left")
    df.to_csv(out / "comparison.csv", index=False)

    n_groups = df.city_cluster.nunique()
    lines = [f"# Network groups vs classical measures ({mode} mode)", ""]
    lines += [f"{len(df)} cities, {n_groups} network group(s).", ""]
    if len(df) < 10:
        lines += ["**Caution:** with fewer than 10 cities these numbers can easily happen by "
                  "chance. Treat them as a check that the pipeline works, not as a finding.", ""]

    ct = pd.crosstab(df.city_cluster, df.get("classical_type", pd.Series(dtype=str)).fillna("missing"))
    lines += ["## Network group vs classical type", "", md_table(ct), ""]

    s = scores(df.city_cluster, df.get("classical_type"))
    lines += ["## Agreement with the classical tests", "",
              f"- Cities with a clear classical type: {s['cities']}",
              f"- Best-match agreement: {s['best_match']}  (share of cities the group's majority type gets right)",
              f"- Adjusted Rand index: {s['ari']}  (0 = chance, 1 = identical)", ""]
    result = {"classical": s}

    if "true_type" in df:
        t = scores(df.city_cluster, df.true_type)
        k = scores(df.classical_type.fillna("unclear"), df.true_type) if "classical_type" in df else {}
        lines += ["## Against the known answer (fake cities)", "",
                  f"- Network groups: best-match {t['best_match']}, ARI {t['ari']}",
                  f"- Classical tests: best-match {k.get('best_match')}, ARI {k.get('ari')}", ""]
        result["truth_network"], result["truth_classical"] = t, k

    measures = [c for c in ["gradient_per_km", "gradient_r2", "share_within_5km", "gs_subcentres",
                            "gs_main_share", "mcmillen_subcentres"] if c in df]
    if measures:
        means = df.groupby("city_cluster")[measures].mean().round(3)
        means.insert(0, "cities", df.groupby("city_cluster").size())
        lines += ["## Classical measures by network group (averages)", "", md_table(means), ""]
        tests = []
        for m in ("gradient_r2", "gs_subcentres", "mcmillen_subcentres"):
            groups = [g[m].dropna().to_numpy() for _, g in df.groupby("city_cluster")]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2 and np.ptp(np.concatenate(groups)) > 0:
                p = kruskal(*groups).pvalue
                tests.append(f"- {m}: Kruskal-Wallis p = {p:.3g}")
        if tests:
            lines += ["Do the groups differ? (small p, e.g. below 0.05, means yes)", ""] + tests + [""]

    rank = ranking(df, out, mode)
    if rank:
        lines += ["## Ranking check", "",
                  "Grouping can cut a smooth spectrum into more pieces than there are 'types', which "
                  "lowers the scores above even when the network has the order right. So each city is "
                  "also given one score: the main direction of variation in what the network learned "
                  "about it (first principal component). If that score ranks cities in the same order "
                  "as a classical measure, the rank correlation is near 1 (the sign does not matter). "
                  "Against a yes/no measure such as the known answer, about 0.87 is already a "
                  "perfect ordering when the two sides are equal in size.", "",
                  "| compared with | rank correlation (abs. Spearman) | p |", "|---|---|---|"]
        lines += [f"| {m} | {r:.2f} | {p:.3g} |" for m, (r, p) in rank.items()]
        lines += [""]
        result["ranking"] = {m: r for m, (r, _) in rank.items()}

    lines += ["## How to read this", "",
              "- High agreement means the network has rediscovered the monocentric/polycentric "
              "split from the layouts alone.",
              "- Low agreement is not automatically bad: the network may be sorting cities by "
              "something else (look at cluster_profiles.png to see what each group contains). "
              "It may also be sorting by how well each city is mapped, which is the risk to rule out.",
              "- The classical tests here use POI counts instead of jobs and relative cut-offs; see "
              "analysis/classical.py for the exact rules.", ""]
    (out / "comparison_report.md").write_text("\n".join(lines))
    plot(out, df, measures)
    say("\n".join(lines[:4] + [f"Agreement with classical tests: best-match {s['best_match']}, ARI {s['ari']}"]
                  + ([f"Against known answer: network {result['truth_network']}, "
                      f"classical {result['truth_classical']}"] if truth else [])))
    say(f"Report -> {out / 'comparison_report.md'}")
    return df, result


def plot(out, df, measures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [m for m in ("gradient_r2", "gs_subcentres", "mcmillen_subcentres") if m in measures]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), squeeze=False)
    groups = sorted(df.city_cluster.unique())
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[0], show):
        for i, g in enumerate(groups):
            v = df.loc[df.city_cluster == g, m].dropna()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=30, alpha=0.8)
            if len(v):
                ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1.5)
        ax.set_xticks(range(len(groups)), [f"group {g}" for g in groups])
        ax.set_title(m.replace("_", " "), fontsize=10)
    fig.suptitle("Classical measures for each network group (one dot per city, bar = mean)", fontsize=10)
    fig.savefig(out / "comparison.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--mode", choices=["city", "tile"], required=True)
    ap.add_argument("--truth", default=None, help="optional CSV with city,true_type")
    a = ap.parse_args(argv)
    run(a.data_dir, a.mode, a.truth)


if __name__ == "__main__":
    main()
