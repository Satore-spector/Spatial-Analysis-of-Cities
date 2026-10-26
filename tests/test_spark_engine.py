"""Check the Spark options give the same results as the normal ones.

Needs Java 17+ and pyspark (see analysis/spark_utils.py). About a minute.

    python tests/test_spark_engine.py
    python run_analysis.py --test-spark

1. Step 2: rasterize 6 fake cities with numpy and with Spark SQL; every grid
   must be identical.
2. Step 4: Spark MLlib k-means on three well-separated blobs of points must
   find the three blobs.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def check(name, ok, detail=""):
    print(f"  {name:58s} {'ok' if ok else 'FAILED'} {detail}", flush=True)
    if not ok:
        raise AssertionError(f"Test failed: {name} {detail}")


def main(argv=None):
    import make_fake_cities
    from analysis import rasterize
    from analysis.cluster import spark_group
    from analysis.common import load_city_grid
    from analysis.spark_utils import get_spark, stop_spark
    from sklearn.metrics import adjusted_rand_score

    print("Starting Spark", flush=True)
    get_spark()
    work = Path(tempfile.mkdtemp(prefix="spark_test_"))
    try:
        data = work / "data"
        make_fake_cities.main(["--out", str(data), "--n", "6", "--seed", "3"])
        rasterize.run(data, engine="pandas", force=True)
        saved = work / "pandas_rasters"
        shutil.copytree(data / "rasters", saved)
        rasterize.run(data, engine="spark", force=True)

        print("\nChecks", flush=True)
        worst = 0.0
        for f in sorted(saved.glob("*.npz")):
            a, ma, _ = load_city_grid(f)
            b, mb, meta = load_city_grid(data / "rasters" / f.name)
            check(f"{f.stem}: same boundary mask", (ma == mb).all())
            worst = max(worst, float(np.abs(a - b).max()))
        check("step 2: Spark grids equal numpy grids", worst < 1e-6, f"(max difference {worst:g})")
        check("step 2: grids say they were made by Spark", meta.get("engine") == "spark")

        rng = np.random.default_rng(0)
        centres = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
        truth = np.repeat([0, 1, 2], 60)
        z = centres[truth] + rng.normal(0, 1, (len(truth), 3))
        labels, info = spark_group(z, None, 6, 0)
        ari = adjusted_rand_score(truth, labels)
        check("step 4: Spark MLlib k-means finds the 3 blobs", info["k"] == 3 and ari > 0.95,
              f"(k={info['k']}, ARI {ari:.2f})")
        print("\nSPARK TESTS PASSED", flush=True)
        return 0
    finally:
        stop_spark()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
