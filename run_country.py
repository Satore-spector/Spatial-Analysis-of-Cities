"""Classify every city in a country: city list -> OSM -> GHSL -> analysis -> types.

    python run_country.py --country India --project MY_GCP_PROJECT

Stages (each can be skipped, and each picks up where it stopped if rerun):
  list      build_city_list.py      GHSL urban centres with more than --min-pop people
                                    (default 100,000), one consistent boundary each
  osm       osm_batch_extract.py    POIs, buildings, roads for every city, reading
                                    each Geofabrik regional file once
  ghsl      ghsl_builtup_100m.py    GHSL built-up surface at 100 m from Earth Engine
  analysis  steps 2-5               grids, classical tests on GHSL and on POIs,
                                    autoencoder + groups (tile and city mode)
  classify  analysis/classify.py    one type per city, map, report

Everything goes into --data-dir (default data/<country>), so different
countries and the earlier single-city runs stay apart. The downloaded UCDB
and OSM files are kept in data/uc and data/osm_pbf and shared.

Results: <data-dir>/analysis/classification/report.md, city_types.csv,
city_types.gpkg (open in QGIS) and city_types_map.png.

Examples:
    python run_country.py --country India --project my-gcp              everything
    python run_country.py --country India --skip-ghsl                   OSM only (no Earth Engine)
    python run_country.py --country India --only list osm               just these stages
    python run_country.py --country India --project my-gcp --limit 20   the 20 largest cities first
    python run_country.py --country India --engine spark                Spark for steps 2 and 4
    python run_country.py --test                                        offline test on fake cities
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STAGES = ["list", "osm", "ghsl", "analysis", "classify"]


def say(msg=""):
    print(msg, flush=True)


def country_code(country: str) -> str:
    sys.path.insert(0, str(ROOT / "data_extraction"))
    from build_city_list import country_code as cc
    return cc(country)


def script(name, args):
    """Run one of the data_extraction scripts with this Python; True if it worked."""
    cmd = [sys.executable, str(ROOT / "data_extraction" / name)] + [str(a) for a in args]
    say("$ " + " ".join(cmd[1:]))
    return subprocess.call(cmd, cwd=ROOT) == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="India")
    ap.add_argument("--min-pop", type=float, default=100_000)
    ap.add_argument("--data-dir", type=Path, default=None, help="default data/<country>")
    ap.add_argument("--only", nargs="*", choices=STAGES, help="run only these stages")
    for s in STAGES:
        ap.add_argument(f"--skip-{s}", action="store_true")
    ap.add_argument("--limit", type=int, help="only the N largest cities (for a first test)")
    g = ap.add_argument_group("list")
    g.add_argument("--ucdb", type=Path, help="GHS-UCDB file you already downloaded")
    g = ap.add_argument_group("osm")
    g.add_argument("--zones", nargs="*", help="Geofabrik zones (default: all of the country)")
    g.add_argument("--pbf", nargs="*", type=Path, help=".osm.pbf files to use instead of downloading")
    g = ap.add_argument_group("ghsl")
    g.add_argument("--project", help="Google Cloud project for Earth Engine (or env EE_PROJECT)")
    g.add_argument("--epoch", type=int, default=2020)
    g.add_argument("--ee-workers", type=int, default=4)
    g = ap.add_argument_group("analysis")
    g.add_argument("--modes", nargs="*", default=["tile", "city"], choices=["tile", "city"])
    g.add_argument("--network-layers", choices=["ghsl", "all"], default="ghsl",
                   help="what the network learns from: 'ghsl' = GHSL built-up + roads (default; least "
                        "affected by uneven OSM mapping), 'all' = every layer incl. POIs and buildings")
    g.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    g.add_argument("--workers", type=int, default=1, help="cities to grid in parallel (step 2)")
    g.add_argument("--cell", type=float, default=100)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--force", action="store_true", help="redo grids that already exist")
    g.add_argument("--no-maps", action="store_true", help="skip per-city classical map images")
    ap.add_argument("--test", action="store_true", help="run the offline test on fake cities and exit")
    a = ap.parse_args(argv)
    if a.test:
        sys.path.insert(0, str(ROOT / "tests"))
        import test_country_offline
        return test_country_offline.main([])

    cc = country_code(a.country)
    data = a.data_dir or Path("data") / cc
    csv = data / "uc" / f"{cc}_cities.csv"
    run = set(a.only) if a.only else {s for s in STAGES if not getattr(a, f"skip_{s}")}
    say(f"Country: {a.country}   data: {data}   stages: {', '.join(s for s in STAGES if s in run)}")
    log, t_all = [], time.time()

    def done(stage, ok, t0):
        log.append((stage, "ok" if ok else "FAILED", (time.time() - t0) / 60))
        if not ok:
            say(f"\n{stage} did not finish; fix the message above and rerun (finished work is kept).")
        return ok

    # list
    if "list" in run:
        t0 = time.time()
        args = ["--country", a.country, "--min-pop", a.min_pop, "--data-dir", data]
        if a.ucdb:
            args += ["--ucdb", a.ucdb]
        if not done("list", script("build_city_list.py", args), t0):
            return finish(log, t_all)
    if not csv.exists():
        say(f"No city list at {csv}. Run the 'list' stage first.")
        return 1
    cities = pd.read_csv(csv)
    if a.limit:
        cities = cities.head(a.limit)
    slugs = cities["slug"].tolist()
    say(f"{len(slugs)} cities in the run")

    # osm
    if "osm" in run:
        t0 = time.time()
        args = ["--cities-csv", csv, "--data-dir", data]
        if a.zones:
            args += ["--zones"] + a.zones
        if a.pbf:
            args += ["--pbf"] + a.pbf
        if a.limit:
            args += ["--only"] + slugs
        if not done("osm", script("osm_batch_extract.py", args), t0):
            return finish(log, t_all)

    # ghsl
    if "ghsl" in run:
        t0 = time.time()
        import os
        project = a.project or os.environ.get("EE_PROJECT")
        if not project:
            say("No Earth Engine project (--project or EE_PROJECT): skipping GHSL. "
                "Types will then rest on OSM only.")
            log.append(("ghsl", "skipped", 0))
        else:
            args = ["--project", project, "--cities-csv", csv, "--data-dir", data,
                    "--epoch", a.epoch, "--workers", a.ee_workers]
            if a.limit:
                args += ["--only"] + slugs
            done("ghsl", script("ghsl_builtup_100m.py", args), t0)   # failures are retried on rerun

    # analysis
    if "analysis" in run:
        from analysis import classical, cluster, compare, rasterize, samples, train
        sel = slugs if a.limit else None

        def step(name, fn):
            t0 = time.time()
            say("\n" + "#" * 70 + f"\n# {name}\n" + "#" * 70)
            try:
                fn()
                ok = True
            except SystemExit as e:
                ok = e.code in (None, 0)
                if not ok:
                    say(f"{name} stopped: {e}")
            log.append((name, "ok" if ok else "FAILED", (time.time() - t0) / 60))
            return ok

        step("grids (step 2)", lambda: rasterize.run(data, sel, a.cell, "both", a.force, a.engine, a.workers))
        ghsl_n = len(list((data / "ghsl").glob("*_ghs_built_s_*.tif"))) if (data / "ghsl").exists() else 0
        measure = "built_up" if ghsl_n else "pois"
        if ghsl_n:
            step("classical tests on GHSL built-up", lambda: classical.run(data, sel, "built_up", maps=not a.no_maps))
        step("classical tests on OSM POIs", lambda: classical.run(data, sel, "pois", maps=not a.no_maps))
        method = "spark-kmeans" if a.engine == "spark" else "kmeans"
        channels = "auto"
        if a.network_layers == "ghsl":
            if ghsl_n:
                channels = "built_up,roads_major,roads_minor"
            else:
                say("No GHSL rasters, so the network learns from all OSM layers (--network-layers all).")
        for mode in a.modes:
            if not step(f"samples ({mode})", lambda: samples.build(data, mode, sel, channels=channels)):
                continue
            if not step(f"train ({mode})", lambda: train.train(data, mode, a.epochs)):
                continue
            if not step(f"groups ({mode})", lambda: cluster.run(data, mode, method)):
                continue
            step(f"compare ({mode})", lambda: compare.run(data, mode, None, measure))
        if a.engine == "spark":
            from analysis.spark_utils import stop_spark
            stop_spark()

    # classify
    if "classify" in run:
        from analysis import classify
        t0 = time.time()
        try:
            classify.run(data, csv)
            done("classify", True, t0)
        except SystemExit as e:
            say(str(e))
            done("classify", False, t0)
    return finish(log, t_all, data)


def finish(log, t_all, data=None) -> int:
    say("\n" + "=" * 70 + "\nSUMMARY")
    for name, status, mins in log:
        say(f"  {name:38s} {status:8s} {mins:6.1f} min")
    say(f"  total {(time.time() - t_all) / 60:.1f} min")
    if data is not None:
        say(f"  results: {data / 'analysis' / 'classification' / 'report.md'}")
    return 1 if any(s == "FAILED" for _, s, _ in log) else 0


if __name__ == "__main__":
    sys.exit(main())
