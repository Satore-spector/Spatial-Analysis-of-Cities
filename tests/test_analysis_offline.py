"""Offline test of the Python pipeline (steps 2-5) on fake cities.

No downloads and no real data: it makes 16 fake cities (8 with one centre, 8
with three or four), runs every step in both modes, and checks the results
against the known answer. Takes a few minutes on a laptop.

Run from the project root:
    python tests/test_analysis_offline.py
    python run_analysis.py --test
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


def check(name, ok, detail=""):
    print(f"  {name:62s} {'ok' if ok else 'FAILED'} {detail}", flush=True)
    if not ok:
        raise AssertionError(f"Test failed: {name} {detail}")


def test_poi_categories():
    """The Python categories must match amenitiy_wrap.r, including rule order."""
    from analysis.common import poi_category
    cases = [  # (tags, expected)  - order-sensitive cases included
        ({"amenity": "restaurant", "shop": "bakery"}, "Food & drink"),
        ({"shop": "supermarket", "office": "company"}, "Retail"),
        ({"amenity": "bank", "office": "financial"}, "Finance"),
        ({"office": "it"}, "Offices"),
        ({"healthcare": "laboratory"}, "Health"),
        ({"amenity": "pharmacy", "shop": "chemist"}, "Retail"),
        ({"amenity": "coaching"}, "Education"),
        ({"amenity": "place_of_worship"}, "Worship"),
        ({"railway": "station"}, "Transport"),
        ({"amenity": "parking"}, "Transport"),
        ({"leisure": "park"}, "Leisure & culture"),
        ({"amenity": "marketplace"}, "Leisure & culture"),
        ({"amenity": "toilets"}, "Other"),
    ]
    cols = ["amenity", "shop", "office", "public_transport", "railway", "tourism", "leisure", "healthcare"]
    df = pd.DataFrame([{c: t.get(c) for c in cols} for t, _ in cases])
    got = poi_category(df).tolist()
    want = [w for _, w in cases]
    check("POI categories match amenitiy_wrap.r", got == want,
          "" if got == want else str([(g, w) for g, w in zip(got, want) if g != w]))


def test_rotation_invariance():
    import torch
    from analysis.model import ConvAutoencoder, d4
    torch.manual_seed(0)
    m = ConvAutoencoder(5, 32, latent=8).eval()
    x = torch.rand(3, 5, 32, 32)
    base = m.embed_invariant(x)
    worst = max(float((m.embed_invariant(d4(x, k)) - base).abs().max()) for k in range(8))
    check("embedding is the same for every rotation/flip", worst < 1e-4, f"(max diff {worst:.1e})")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    keep = "--keep" in argv
    print("Unit checks", flush=True)
    test_poi_categories()
    test_rotation_invariance()

    import make_fake_cities
    import run_analysis

    work = Path(tempfile.mkdtemp(prefix="fake_cities_"))
    data = work / "data"
    try:
        print(f"\nMaking 16 fake cities in {data}", flush=True)
        make_fake_cities.main(["--out", str(data), "--n", "16", "--seed", "11"])
        truth = data / "fake_cities_truth.csv"
        status = run_analysis.main(["--data-dir", str(data), "--mode", "both", "--truth", str(truth),
                                    "--tile-stride", "16", "--epochs", "40", "--seed", "0"])
        print("\nChecks", flush=True)
        check("pipeline finished without failed steps", status == 0)
        t = pd.read_csv(truth)
        n = len(t)
        check("a grid for every city", len(list((data / "rasters").glob("*.npz"))) == n)

        cl = pd.read_csv(data / "analysis" / "classical" / "classical_measures.csv").merge(t, on="city")
        mono, poly = cl[cl.true_type == "monocentric"], cl[cl.true_type == "polycentric"]
        check("classical: no subcentres in one-centre cities",
              (mono.gs_subcentres == 0).all() and (mono.mcmillen_subcentres == 0).all())
        check("classical: 2+ subcentres in many-centre cities",
              (poly.gs_subcentres >= 2).all() and (poly.mcmillen_subcentres >= 2).all())
        check("classical: subcentres = planted centres - 1 (Giuliano-Small)",
              (poly.gs_subcentres == poly.true_centres - 1).mean() >= 0.75,
              f"({(poly.gs_subcentres == poly.true_centres - 1).sum()}/{len(poly)})")
        check("classical: gradient fits one-centre cities better",
              mono.gradient_r2.mean() > poly.gradient_r2.mean(),
              f"(R2 {mono.gradient_r2.mean():.2f} vs {poly.gradient_r2.mean():.2f})")

        for mode in ("city", "tile"):
            out = data / "analysis" / mode
            emb = pd.read_csv(out / "embeddings.csv")
            smp = pd.read_csv(out / "samples.csv")
            check(f"{mode}: one embedding per sample", len(emb) == len(smp) and emb.notna().all().all(),
                  f"({len(emb)})")
            tr = pd.read_csv(out / "training_loss.csv")
            check(f"{mode}: training reduced the rebuild error",
                  tr.val_loss.min() < tr.val_loss.iloc[0], f"({tr.val_loss.iloc[0]:.4f} -> {tr.val_loss.min():.4f})")
            for f in ("clusters.csv", "city_clusters.csv", "cluster_profiles.png", "embedding_map.png",
                      "comparison_report.md", "comparison.png"):
                check(f"{mode}: wrote {f}", (out / f).exists())
            comp = pd.read_csv(out / "comparison.csv")
            from analysis.compare import scores
            s = scores(comp.city_cluster, comp.true_type)
            print(f"  {mode}: network groups vs known answer: best-match {s['best_match']}, "
                  f"ARI {s['ari']} (information only)", flush=True)
        tile = pd.read_csv(data / "analysis" / "tile" / "comparison.csv")
        from analysis.compare import scores
        s = scores(tile.city_cluster, tile.true_type)
        check("tile mode separates one-centre from many-centre cities", s["best_match"] >= 0.8,
              f"(best-match {s['best_match']})")
        from analysis.compare import ranking
        r = ranking(tile, data / "analysis" / "tile", "tile")
        rho = r.get("known answer (fake cities)", (0, 1))[0]
        check("tile mode ranks cities in the right order", rho >= 0.7, f"(Spearman {rho:.2f}; 0.87 = perfect)")
        print("\nALL TESTS PASSED", flush=True)
        return 0
    finally:
        if keep:
            print(f"Kept fake data and results in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
