Drop city_types.csv here (copy from data/india/analysis/classification/) so the site can load it.
Optional map files (run from the project root, after the pipeline):
  python export_boundaries.py     -> city_boundaries.geojson (outline of each city)
  python export_city_grids.py     -> city_grids/<slug>.json (the Map view: POIs, roads, built-up outlines; plus heat layers and centres)
                                       needs data/<run>/<slug>_pois.gpkg and _roads.gpkg; add --no-osm to skip the Map view
Without city_grids/ the page shows the outline only.
