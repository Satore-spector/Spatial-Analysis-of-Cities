"""Offline test of the country pipeline (run_country.py) on fake cities.

Makes 16 fake cities (8 one-centre, 8 several-centre) at Indian latitudes and
longitudes, a fake GHSL Urban Centre Database with decoys, two real .osm.pbf
files (one city in both, two cities badly mapped), and GHSL-style rasters for
every city in place of the Earth Engine download. Then runs every stage except
the Earth Engine one and checks the results. About 3-5 minutes; no internet.

    python tests/test_country_offline.py
    python run_country.py --test
Add --keep to keep the fake data and results (the folder is printed).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

THIN = ["fakeabpoly", "fakeagmono"]


def check(name, ok, detail=""):
    print(f"  {name:60s} {'ok' if ok else 'FAILED'} {detail}", flush=True)
    if not ok:
        raise AssertionError(f"Test failed: {name} {detail}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    import make_fake_cities
    import make_fake_osm
    import run_country

    work = Path(tempfile.mkdtemp(prefix="fake_country_"))
    src, raw, data = work / "src", work / "raw", work / "data"
    try:
        make_fake_cities.main(["--out", str(src), "--n", "16", "--seed", "11", "--ghsl-every", "1"])
        make_fake_osm.main(["--cities", str(src), "--out", str(raw), "--thin"] + THIN)
        (data / "ghsl").mkdir(parents=True)
        for f in (src / "ghsl").glob("*.tif"):   # stands in for the Earth Engine stage
            shutil.copy(f, data / "ghsl" / f.name)
        status = run_country.main(["--country", "India", "--data-dir", str(data),
                                   "--ucdb", str(raw / "fake_ucdb.gpkg"),
                                   "--pbf", str(raw / "zone-a.osm.pbf"), str(raw / "zone-b.osm.pbf"),
                                   "--skip-ghsl", "--epochs", "40", "--no-maps"])
        print("\nChecks", flush=True)
        check("run finished without failed stages", status == 0)
        truth = pd.read_csv(src / "fake_cities_truth.csv")
        lst = pd.read_csv(data / "uc" / "india_cities.csv")
        check("city list: the 16 cities, decoys left out", sorted(lst.slug) == sorted(truth.city),
              f"({len(lst)} rows)")

        osm = pd.read_csv(data / "osm_extraction_summary.csv").merge(
            pd.read_csv(raw / "fake_osm_truth.csv").rename(columns={"buildings": "true_buildings"}), on="city")
        check("OSM: every POI found once (city in two files counted once)",
              (osm.pois == osm.pois_node + osm.pois_area).all())
        check("OSM: every building and road found once",
              (osm.buildings == osm.true_buildings).all() and (osm.roads == osm.road_ways).all())

        out = data / "analysis" / "classification"
        for f in ("city_types.csv", "city_types.gpkg", "city_types_map.png", "report.md"):
            check(f"classification: wrote {f}", (out / f).exists())
        ct = pd.read_csv(out / "city_types.csv").merge(truth[["city", "true_type"]], on="city")
        check("one row per city", len(ct) == len(truth))
        check("badly mapped cities flagged", set(ct.loc[ct.osm_quality.isin(["sparse", "very sparse"]), "city"]) == set(THIN))
        mono = ct[ct.true_type == "monocentric"]
        check("GHSL tests: no one-centre city called polycentric", (mono.ghsl_type != "polycentric").all())
        g_ok = (ct.ghsl_type == ct.true_type).mean()
        check("GHSL tests: type right for at least 75% of cities", g_ok >= 0.75, f"({g_ok:.0%})")
        clear = ct[ct.final_type.isin(["monocentric", "polycentric"])]
        acc = (clear.final_type == clear.true_type).mean()
        check("final type right for at least 85% of typed cities", acc >= 0.85, f"({acc:.0%} of {len(clear)})")
        high = ct[ct.confidence == "high"]
        check("every 'high' confidence type is right", (high.final_type == high.true_type).all(),
              f"({len(high)} cities)")
        from sklearn.metrics import adjusted_rand_score
        from analysis.compare import best_match
        tile = ct.dropna(subset=["network_group_tile"])
        bm = best_match(tile.network_group_tile, tile.true_type)
        check("network (GHSL + roads, tile mode) separates the two kinds", bm >= 0.8,
              f"(best-match {bm:.2f}, ARI {adjusted_rand_score(tile.true_type, tile.network_group_tile):.2f})")
        rep = (out / "report.md").read_text()
        tile_part = rep.split("## Network groups (tile mode)")[1].split("## ")[0]
        check("report: tile-mode groups do not follow OSM mapping quality", "Warning" not in tile_part)
        print("\nALL COUNTRY TESTS PASSED", flush=True)
        return 0
    finally:
        if "--keep" in argv:
            print(f"Kept fake data and results in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
