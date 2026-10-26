"""Step 4: group similar embeddings and draw the results.

Reads data/analysis/<mode>/embeddings.csv and samples.npz/csv. Writes:

  clusters.csv          one row per sample: its group and its 2-D map position
  cluster_profiles.csv  what each group looks like: average intensity of every
  cluster_profiles.png  layer (offices, retail, roads ...) in that group
  embedding_map.png     every sample as a dot on a 2-D map (UMAP, or PCA for
                        small sets); close dots = similar layouts
  city_clusters.csv     one row per city with its group ("city_cluster"),
                        the input to the comparison in step 5
  tile mode only:
  city_profiles.csv     for each city, the share of its tiles in each group
  tile_maps.png         each city drawn as its tiles, coloured by group

Grouping: k-means. In city mode the number of groups is the one with the
best silhouette score (how cleanly the groups separate), trying 2..--max-k.
In tile mode the default is 8 tile types: tile embeddings usually have no
clear natural number of groups (silhouette is nearly flat), and picking the
"best" then tends to give just 2 types (inner vs outer city), too coarse to
tell one centre from several. Cities are then grouped by their mix of tile
types, with the number of city groups chosen by silhouette.
--k sets the number yourself; --k auto always uses the silhouette.
--method hdbscan finds the number of groups itself and may leave some samples
unassigned (-1).
--method spark-kmeans runs the same k-means in Spark MLlib instead of
scikit-learn (needs Java + pyspark; see analysis/spark_utils.py).

Usage:
    python -m analysis.cluster --mode tile
    python -m analysis.cluster --mode city --k 2
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import DATA_DIR, analysis_dir, say, write_json


def group(z: np.ndarray, method: str, k: int | None, max_k: int, seed: int, min_size: int | None):
    """Returns labels and a small dict describing how they were chosen."""
    n = len(z)
    if n < 3:
        return np.zeros(n, int), {"method": "none", "note": "fewer than 3 samples"}
    if method == "spark-kmeans":
        return spark_group(z, k, max_k, seed)
    if method == "hdbscan":
        m = min_size or max(5, n // 50)
        labels = HDBSCAN(min_cluster_size=m).fit_predict(z)
        return labels, {"method": "hdbscan", "min_cluster_size": m,
                        "groups": int(labels.max() + 1), "unassigned": int((labels < 0).sum())}
    if k:
        k = min(k, n - 1)
        km = KMeans(k, n_init=20, random_state=seed).fit(z)
        sil = silhouette_score(z, km.labels_) if 1 < k < n else np.nan
        return km.labels_, {"method": "kmeans", "k": k, "silhouette": sil}
    scores = {}
    for kk in range(2, min(max_k, n - 1) + 1):
        lab = KMeans(kk, n_init=20, random_state=seed).fit_predict(z)
        scores[kk] = silhouette_score(z, lab, sample_size=min(n, 5000), random_state=seed)
    best = max(scores, key=scores.get)
    say("  silhouette by number of groups: " +
        ", ".join(f"{kk}: {s:.2f}" for kk, s in scores.items()) + f"  -> using {best}")
    labels = KMeans(best, n_init=20, random_state=seed).fit_predict(z)
    return labels, {"method": "kmeans", "k": best, "silhouette": scores[best],
                    "silhouette_by_k": scores}


def spark_group(z: np.ndarray, k: int | None, max_k: int, seed: int):
    """k-means with Spark MLlib (pyspark.ml.clustering.KMeans).

    Same idea as the scikit-learn version: try 2..max_k groups and keep the
    best silhouette (Spark's ClusteringEvaluator) unless k is given. Spark's
    silhouette uses squared distances, so its numbers differ a little from
    scikit-learn's.
    """
    from pyspark.ml.clustering import KMeans as SparkKMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import VectorAssembler

    from .spark_utils import get_spark

    spark = get_spark()
    cols = [f"f{i}" for i in range(z.shape[1])]
    pdf = pd.DataFrame(z, columns=cols)
    pdf.insert(0, "row_id", np.arange(len(z)))
    df = VectorAssembler(inputCols=cols, outputCol="features").transform(spark.createDataFrame(pdf)).cache()
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction")

    def fit(kk):
        model = SparkKMeans(k=kk, seed=seed, maxIter=100, featuresCol="features").fit(df)
        pred = model.transform(df)
        return pred, evaluator.evaluate(pred)

    n = len(z)
    if k:
        k = min(k, n - 1)
        pred, sil = fit(k)
        info = {"method": "spark-kmeans", "k": k, "silhouette_spark": sil}
    else:
        scores, preds = {}, {}
        for kk in range(2, min(max_k, n - 1) + 1):
            preds[kk], scores[kk] = fit(kk)
        k = max(scores, key=scores.get)
        say("  Spark silhouette by number of groups: " +
            ", ".join(f"{kk}: {s:.2f}" for kk, s in scores.items()) + f"  -> using {k}")
        pred = preds[k]
        info = {"method": "spark-kmeans", "k": k, "silhouette_spark": scores[k],
                "silhouette_by_k": scores}
    labels = pred.select("row_id", "prediction").toPandas().sort_values("row_id")["prediction"].to_numpy()
    df.unpersist()
    return labels.astype(int), info


def to_2d(z: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if len(z) >= 20:
        try:
            import umap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xy = umap.UMAP(n_neighbors=min(15, len(z) - 1), random_state=seed).fit_transform(z)
            return xy, "UMAP"
        except ImportError:
            say("  (umap-learn not installed; using PCA for the 2-D map)")
    if len(z) < 2:
        return np.zeros((len(z), 2)), "none"
    return PCA(2, random_state=seed).fit_transform(z), "PCA"


TILE_TYPES_DEFAULT = 8


def resolve_k(k, mode):
    """--k: a number, 'auto' (silhouette) or unset (8 in tile mode, auto in city mode)."""
    if k in (None, ""):
        return TILE_TYPES_DEFAULT if mode == "tile" else None
    if str(k).lower() == "auto":
        return None
    return int(k)


def run(data_dir, mode, method="kmeans", k=None, max_k=8, city_k=None, seed=0, min_size=None):
    out = analysis_dir(data_dir, mode)
    if not (out / "embeddings.csv").exists():
        raise SystemExit(f"No embeddings in {out}. Run: python -m analysis.train --mode {mode}")
    emb = pd.read_csv(out / "embeddings.csv")
    meta = pd.read_csv(out / "samples.csv")
    with np.load(out / "samples.npz") as zf:
        x, channels = zf["x"], list(zf["channels"])
    zcols = [c for c in emb.columns if c.startswith("z")]
    z = StandardScaler().fit_transform(emb[zcols].to_numpy())

    say(f"Grouping {len(z)} {mode} samples")
    k = resolve_k(k, mode)
    labels, info = group(z, method, k, max_k, seed, min_size)
    xy, how2d = to_2d(z, seed)
    res = meta.merge(emb[["sample_id"]], on="sample_id")
    res["cluster"] = labels
    res["map_x"], res["map_y"] = xy[:, 0], xy[:, 1]
    res.to_csv(out / "clusters.csv", index=False)

    # What each group looks like: average of each (normalised) layer.
    layer_means = x.mean(axis=(2, 3))
    prof = pd.DataFrame(layer_means, columns=channels)
    prof["cluster"] = labels
    profiles = prof.groupby("cluster").mean()
    profiles.insert(0, "n_samples", prof.groupby("cluster").size())
    if mode == "tile":
        profiles.insert(1, "mean_dist_km", res.groupby("cluster")["dist_km"].mean().round(2))
    profiles.round(4).to_csv(out / "cluster_profiles.csv")
    say("  groups: " + ", ".join(f"{c}: {n}" for c, n in profiles["n_samples"].items()))

    if mode == "city":
        cities = res[["city", "cluster"]].rename(columns={"cluster": "city_cluster"})
        city_info = info
    else:
        cities, city_info = city_level_from_tiles(res, out, city_k, max_k, seed, method)
    cities.to_csv(out / "city_clusters.csv", index=False)
    write_json(out / "cluster_settings.json", {"samples": info, "cities": city_info, "map": how2d})
    plots(out, mode, res, profiles, channels, how2d)
    return res, cities


def city_level_from_tiles(res, out, city_k, max_k, seed, method="kmeans"):
    """Describe each city by its mix of tile groups, then group the cities."""
    share = pd.crosstab(res["city"], res["cluster"], normalize="index")
    share.columns = [f"share_group_{c}" for c in share.columns]
    # Where each tile group sits: its average distance from the city's centre.
    dist = res.pivot_table(index="city", columns="cluster", values="dist_km", aggfunc="mean")
    dist.columns = [f"dist_km_group_{c}" for c in dist.columns]
    profiles = share.join(dist).round(4)
    profiles.to_csv(out / "city_profiles.csv")
    n = len(share)
    if n < 3:
        say("  fewer than 3 cities: not grouping cities (see city_profiles.csv)")
        return pd.DataFrame({"city": share.index, "city_cluster": 0}), {"note": "fewer than 3 cities"}
    say("Grouping cities by their mix of tile groups")
    z = share.to_numpy()
    labels, info = group(z, "spark-kmeans" if method == "spark-kmeans" else "kmeans",
                         city_k, min(max_k, 6), seed, None)
    return pd.DataFrame({"city": share.index, "city_cluster": labels}), info


def plots(out, mode, res, profiles, channels, how2d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")

    def colour(c):
        return (0.75, 0.75, 0.75, 1) if c < 0 else cmap(c % 10)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(res.map_x, res.map_y, c=[colour(c) for c in res.cluster],
               s=40 if mode == "city" else 6, alpha=0.9 if mode == "city" else 0.6)
    if mode == "city":
        for _, r in res.iterrows():
            ax.annotate(r.city, (r.map_x, r.map_y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for c in sorted(res.cluster.unique()):
        ax.scatter([], [], color=colour(c), label=f"group {c}" if c >= 0 else "unassigned")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{mode} embeddings ({how2d}); close = similar layout", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out / "embedding_map.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    p = profiles[channels]
    rel = p / p.max(axis=0).replace(0, 1)  # each layer scaled to its busiest group
    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(channels), 1.2 + 0.45 * len(p)))
    ax.imshow(rel.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(channels)), [c.replace("poi_", "") for c in channels],
                  rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(p)), [f"group {c} (n={n})" for c, n in profiles["n_samples"].items()],
                  fontsize=8)
    ax.set_title("What each group contains (darker = more of that layer)", fontsize=10)
    fig.savefig(out / "cluster_profiles.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    if mode == "tile":
        cities = sorted(res.city.unique())
        ncol = min(4, len(cities))
        nrow = int(np.ceil(len(cities) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, city in zip(axes.flat, cities):
            t = res[res.city == city]
            size = int(t.size_cells.iloc[0])
            for _, r in t.iterrows():
                ax.add_patch(plt.Rectangle((r.col0, -r.row0 - size), size, size,
                                           color=colour(r.cluster), alpha=0.55, lw=0))
            ax.set_xlim(t.col0.min(), t.col0.max() + size)
            ax.set_ylim(-(t.row0.max() + size), -t.row0.min())
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(city, fontsize=9)
        for ax in list(axes.flat)[len(cities):]:
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=colour(c)) for c in sorted(res.cluster.unique())]
        fig.legend(handles, [f"group {c}" for c in sorted(res.cluster.unique())],
                   loc="lower center", ncol=min(8, len(handles)), frameon=False, fontsize=8)
        fig.savefig(out / "tile_maps.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def add_args(ap):
    ap.add_argument("--method", choices=["kmeans", "hdbscan", "spark-kmeans"], default="kmeans",
                    help="spark-kmeans: k-means with Spark MLlib (needs Java + pyspark)")
    ap.add_argument("--k", default=None,
                    help="number of groups, or 'auto' for best silhouette "
                         f"(default: {TILE_TYPES_DEFAULT} in tile mode, auto in city mode)")
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--city-k", type=int, default=None,
                    help="tile mode: number of city groups (default: best silhouette)")
    ap.add_argument("--min-cluster-size", type=int, default=None, help="hdbscan only")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--mode", choices=["city", "tile"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    add_args(ap)
    a = ap.parse_args(argv)
    run(a.data_dir, a.mode, a.method, a.k, a.max_k, a.city_k, a.seed, a.min_cluster_size)


if __name__ == "__main__":
    main()
