"""Run the Python half of the pipeline (steps 2-5) in one go.

Run the R extraction first (run_pipeline.R or data_extraction_osm.R) so data/
holds <city>_boundary/pois/buildings/roads.gpkg files. Then, from the project root:

    python run_analysis.py                       both modes, every city in data/
    python run_analysis.py --mode tile           just patches inside cities
    python run_analysis.py --mode city bengaluru pune delhi
    python run_analysis.py --test                offline test on fake cities (no data needed)
    python run_analysis.py --test-spark          check Spark gives the same results
    python run_analysis.py --engine spark        use Spark for step 2 (Spark SQL) and
                                                 step 4 (Spark MLlib k-means)

Steps:
    2  rasterize   city .gpkg files -> grids            (data/rasters/)
    3  samples     grids -> training examples            (data/analysis/<mode>/)
       train       autoencoder -> one embedding each
    4  cluster     embeddings -> groups, maps
    5  classical   rule-based subcentre tests            (data/analysis/classical/)
       compare     do the groups match the tests?        (comparison_report.md)

Every option of the individual steps can be passed here too (see --help);
--skip-rasterize / --skip-train etc. reuse earlier outputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from analysis import classical, cluster, compare, rasterize, samples, train
from analysis.common import DATA_DIR, say


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cities", nargs="*", help="city names or slugs (default: all in data/)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--mode", choices=["city", "tile", "both"], default="both")
    ap.add_argument("--test", action="store_true", help="run the offline fake-city test and exit")
    ap.add_argument("--test-spark", action="store_true",
                    help="check the Spark options match the normal ones (needs Java + pyspark)")
    for s in ("rasterize", "train", "classical"):
        ap.add_argument(f"--skip-{s}", action="store_true", help=f"reuse existing {s} outputs")
    ap.add_argument("--truth", default=None, help="CSV with city,true_type (fake-city test)")
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas",
                    help="spark: Spark SQL for step 2 and Spark MLlib k-means for step 4 "
                         "(needs Java 17+ and pyspark)")
    g = ap.add_argument_group("step 2 (rasterize)")
    g.add_argument("--cell", type=float, default=100)
    g.add_argument("--poi-source", choices=["both", "node", "area"], default="both")
    g.add_argument("--force", action="store_true", help="redo grids that already exist")
    samples.add_args(ap.add_argument_group("step 3 (samples)"))
    train.add_args(ap.add_argument_group("step 3 (training)"))
    cluster.add_args(ap.add_argument_group("step 4 (clustering)"))
    classical.add_args(ap.add_argument_group("step 5 (classical tests)"))
    a = ap.parse_args(argv)

    if a.test:
        sys.path.insert(0, str(Path(__file__).parent / "tests"))
        import test_analysis_offline
        return test_analysis_offline.main()
    if a.test_spark:
        sys.path.insert(0, str(Path(__file__).parent / "tests"))
        import test_spark_engine
        return test_spark_engine.main()

    log, t_all = [], time.time()
    method = "spark-kmeans" if a.engine == "spark" and a.method == "kmeans" else a.method

    def step(name, fn):
        t0 = time.time()
        say("\n" + "#" * 70 + f"\n# {name}\n" + "#" * 70)
        try:
            fn()
            log.append((name, "ok", time.time() - t0))
            return True
        except SystemExit as e:
            if e.code not in (None, 0):
                say(f"{name} stopped: {e}")
                log.append((name, "FAILED", time.time() - t0))
                return False
            log.append((name, "ok", time.time() - t0))
            return True

    ok = a.skip_rasterize or step(f"Step 2: rasterize ({a.engine})", lambda: rasterize.run(
        a.data_dir, a.cities, a.cell, a.poi_source, a.force, a.engine))
    if not a.skip_classical and ok:
        step("Step 5a: classical tests", lambda: classical.run(
            a.data_dir, a.cities, a.measure, a.vcell, a.density_pct, a.min_share, a.z,
            a.min_dist_km, a.lowess_frac))

    modes = ["city", "tile"] if a.mode == "both" else [a.mode]
    for mode in modes if ok else []:
        if not a.skip_train:
            if not step(f"Step 3: samples ({mode})", lambda: samples.build(
                    a.data_dir, mode, a.cities, a.city_px, a.city_window_km, a.tile_px, a.tile_stride,
                    a.min_inside, a.min_activity, a.normalise, a.channels, a.smooth_m, a.activity)):
                continue
            if not step(f"Step 3: train ({mode})", lambda: train.train(
                    a.data_dir, mode, a.epochs, a.latent, a.batch, a.lr, patience=a.patience,
                    seed=a.seed, device=a.device)):
                continue
        if not step(f"Step 4: cluster ({mode}, {method})", lambda: cluster.run(
                a.data_dir, mode, method, a.k, a.max_k, a.city_k, a.seed, a.min_cluster_size)):
            continue
        step(f"Step 5b: compare ({mode})", lambda: compare.run(a.data_dir, mode, a.truth))

    if a.engine == "spark":
        from analysis.spark_utils import stop_spark
        stop_spark()
    say("\n" + "=" * 70 + "\nSUMMARY")
    for name, status, secs in log:
        say(f"  {name:32s} {status:7s} {secs / 60:5.1f} min")
    say(f"  total {(time.time() - t_all) / 60:.1f} min. Results in {Path(a.data_dir) / 'analysis'}")
    return 1 if any(s == "FAILED" for _, s, _ in log) else 0


if __name__ == "__main__":
    sys.exit(main())
