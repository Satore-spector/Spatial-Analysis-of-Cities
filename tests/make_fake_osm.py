"""Turn fake cities (from make_fake_cities.py) into the raw inputs of the country pipeline.

Writes, so that build_city_list.py and osm_batch_extract.py can be tested
without downloading anything:

  <out>/fake_ucdb.gpkg          a small stand-in for the GHSL Urban Centre Database,
                                with the same column names: the fake cities (country
                                "India") plus decoys (too small, or in another country)
  <out>/zone-a.osm.pbf          real OpenStreetMap files: POIs as nodes, some POIs and
  <out>/zone-b.osm.pbf          a multipolygon relation as areas, buildings as closed
                                ways, roads as ways. The cities are split between the
                                two files, and the first city is in both (as at the
                                edge of two Geofabrik zones), to test de-duplication.
  <out>/fake_osm_truth.csv      what the extractor should find per city

--thin <slug> ... keeps only 10% of those cities' POIs (a badly mapped city).

    python tests/make_fake_osm.py --cities <dir with fake city gpkg files> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
from osmium.osm.mutable import Node, Relation, Way

POI_COLS = ["amenity", "shop", "office", "public_transport", "railway", "tourism", "leisure", "healthcare"]


class Ids:
    def __init__(self):
        self.node, self.way, self.rel = 1, 1, 1

    def n(self):
        self.node += 1
        return self.node

    def w(self):
        self.way += 1
        return self.way

    def r(self):
        self.rel += 1
        return self.rel


def ring_nodes(coords, ids, nodes):
    """Node ids for a closed ring (first node repeated at the end)."""
    pts = list(coords)[:-1]
    refs = []
    for x, y in pts:
        i = ids.n()
        nodes.append(Node(id=i, location=(float(x), float(y)), tags={}))
        refs.append(i)
    return refs + [refs[0]]


def city_objects(slug, src: Path, ids: Ids, rng, thin: bool):
    """OSM nodes/ways/relations for one fake city, plus the expected counts."""
    pois = gpd.read_file(src / f"{slug}_pois.gpkg")
    blds = gpd.read_file(src / f"{slug}_buildings.gpkg")
    roads = gpd.read_file(src / f"{slug}_roads.gpkg")
    boundary = gpd.read_file(src / f"{slug}_boundary.gpkg").union_all()
    if thin:
        pois = pois.sample(frac=0.1, random_state=0)
    nodes, ways, rels = [], [], []
    n_node = n_area = 0
    for row in pois.itertuples():
        tags = {c: getattr(row, c) for c in POI_COLS if isinstance(getattr(row, c), str)}
        x, y = row.geometry.x, row.geometry.y
        if row.poi_source == "area":  # a small square area tagged as the POI
            d = 0.0002
            refs = ring_nodes([(x - d, y - d), (x + d, y - d), (x + d, y + d), (x - d, y + d), (x - d, y - d)],
                              ids, nodes)
            ways.append(Way(id=ids.w(), nodes=refs, tags=tags))
            n_area += 1
        else:
            nodes.append(Node(id=ids.n(), location=(x, y), tags=tags))
            n_node += 1
    # One multipolygon relation (a hospital campus) near the middle of the city.
    c = boundary.representative_point()
    d = 0.002
    outer = ring_nodes([(c.x - d, c.y - d), (c.x + d, c.y - d), (c.x + d, c.y + d), (c.x - d, c.y + d),
                        (c.x - d, c.y - d)], ids, nodes)
    wid = ids.w()
    ways.append(Way(id=wid, nodes=outer, tags={}))
    rels.append(Relation(id=ids.r(), members=[("w", wid, "outer")],
                         tags={"type": "multipolygon", "amenity": "hospital", "name": "Campus"}))
    n_area += 1
    for g in blds.geometry:
        refs = ring_nodes(g.exterior.coords, ids, nodes)
        ways.append(Way(id=ids.w(), nodes=refs, tags={"building": "yes"}))
    for row in roads.itertuples():
        geom = row.geometry
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for ln in lines:
            refs = []
            for x, y in ln.coords:
                i = ids.n()
                nodes.append(Node(id=i, location=(float(x), float(y)), tags={}))
                refs.append(i)
            ways.append(Way(id=ids.w(), nodes=refs, tags={"highway": row.highway}))
    truth = {"city": slug, "pois_node": n_node, "pois_area": n_area, "buildings": len(blds),
             "road_ways": sum(len(r.geometry.geoms) if r.geometry.geom_type == "MultiLineString" else 1
                              for r in roads.itertuples()), "thin": thin}
    return nodes, ways, rels, truth


def write_pbf(path: Path, objs):
    if path.exists():
        path.unlink()
    nodes = sorted((n for o in objs for n in o[0]), key=lambda n: n.id)
    ways = sorted((w for o in objs for w in o[1]), key=lambda w: w.id)
    rels = sorted((r for o in objs for r in o[2]), key=lambda r: r.id)
    with osmium.SimpleWriter(str(path)) as w:
        for n in nodes:
            w.add_node(n)
        for x in ways:
            w.add_way(x)
        for r in rels:
            w.add_relation(r)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thin", nargs="*", default=[])
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    truth = pd.read_csv(args.cities / "fake_cities_truth.csv")
    slugs = truth["city"].tolist()

    # Fake UCDB: one polygon per fake city, plus decoys that must be left out.
    rows = []
    for i, slug in enumerate(slugs):
        g = gpd.read_file(args.cities / f"{slug}_boundary.gpkg").to_crs(4326).union_all()
        rows.append({"ID_UC_G0": 1000 + i, "GC_UCN_MAI_2025": slug.capitalize(), "GC_CNT_GAD_2025": "India",
                     "GC_POP_TOT_2025": float(rng.integers(150_000, 5_000_000)), "geometry": g})
    c = rows[0]["geometry"].centroid
    for j, (name, country, pop) in enumerate([("Smalltown", "India", 60_000), ("Elsewhere", "Nepal", 900_000)]):
        rows.append({"ID_UC_G0": 9000 + j, "GC_UCN_MAI_2025": name, "GC_CNT_GAD_2025": country,
                     "GC_POP_TOT_2025": float(pop), "geometry": c.buffer(0.05 + 0.01 * j)})
    gpd.GeoDataFrame(rows, geometry="geometry", crs=4326).to_file(args.out / "fake_ucdb.gpkg", driver="GPKG")

    ids = Ids()
    objs, trows = {}, []
    for slug in slugs:
        n, w, r, t = city_objects(slug, args.cities, ids, rng, slug in set(args.thin))
        objs[slug] = (n, w, r)
        trows.append(t)
    half = len(slugs) // 2
    zone_a = [objs[s] for s in slugs[:half]]
    zone_b = [objs[s] for s in slugs[half:]] + [objs[slugs[0]]]   # first city in both files
    write_pbf(args.out / "zone-a.osm.pbf", zone_a)
    write_pbf(args.out / "zone-b.osm.pbf", zone_b)
    pd.DataFrame(trows).to_csv(args.out / "fake_osm_truth.csv", index=False)
    print(f"Wrote fake UCDB and 2 OSM files for {len(slugs)} cities to {args.out}")


if __name__ == "__main__":
    main()
