# Spatial Analysis of Cities

A project to find out whether cities can be sorted by their spatial structure (for example **monocentric**: one dominant centre, or **polycentric**: several centres) using patterns learned by a neural network, rather than rules we decide in advance.

This README explains the plan, what exists today, and where help is needed.

## The idea in one paragraph

We download open map data for many cities, turn each city into a set of image-like grids (where the shops, offices, buildings and roads are), and train a neural network **without labels** to learn what makes cities look similar or different. We then cluster what it learned and check whether the groups match known ideas like monocentric and polycentric. The first goal is not a final answer. It is to narrow down broad trends so we know where to look closer.

## Status

| Step | State |
|---|---|
| 1. Download city data from OpenStreetMap | Working for Bengaluru (needs a full test run on more cities) |
| 2. Consistent city boundaries | Partly done (see below) |
| 3. Turn cities into grids | Not started |
| 4. Unsupervised neural network | Not started (will be in Python) |
| 5. Clustering and checking against known measures | Not started |

## How the pipeline will work

1. **Extract** (R, in this repo): for each city, save the boundary, points of interest (POIs), buildings and roads as `.gpkg` files.
2. **Rasterize** (Python, planned): convert each city into a fixed-size grid centred on the city, with one layer (channel) per POI category, plus buildings and roads. Population and night-light layers may be added later.
3. **Learn** (Python, planned): train a convolutional autoencoder (or a contrastive model) that compresses each grid into a short vector. Rotations and flips are treated as the same city, since north should not matter.
4. **Cluster**: group the vectors with k-means or HDBSCAN and view them in 2D with UMAP.
5. **Check**: compare the clusters with classical subcentre-detection methods (for example McMillen, or Giuliano and Small) to see whether the network found something real.

The two languages only meet at the files. R writes `.gpkg`, and Python reads them.

## Repo layout

```
data_extraction/
  data_extraction_osm.R   Main extraction script (boundary, POIs, buildings, roads)
  amenitiy_wrap.r         Groups raw OSM tags into POI categories and plots them
data/                     Inputs and outputs (not committed, see below)
```

## How to run the extraction

Install the R packages once: `osmextract`, `osmdata`, `sf`, `dplyr`, `ggplot2`, and optionally `here`.

From the project root:

```
Rscript data_extraction/data_extraction_osm.R "Bengaluru, India"
Rscript data_extraction/data_extraction_osm.R "Pune, India" "Delhi, India"
Rscript data_extraction/amenitiy_wrap.r bengaluru
```

Useful flags for the extraction script:

- `--force` redo cities that already have output files
- `--strict` stop instead of warning if a boundary falls back to a rough box

What you get in `data/` for each city (`<slug>` is the lower-case city name):

- `<slug>_boundary.gpkg`, `<slug>_pois.gpkg`, `<slug>_buildings.gpkg`, `<slug>_roads.gpkg`
- `extraction_summary.csv`, one row per city with status, counts, boundary source and area in km²
- `osm_pbf/`, cached OpenStreetMap downloads so we do not download the same region twice

Output files are large and are not committed to git. Everyone generates them locally.

## Things to know about the data

- **POIs come from two places.** Some are single points on the map (`poi_source = "node"`), and some are drawn as shapes and converted to one point each (`poi_source = "area"`). Some places may appear twice, once as each. Parks and other leisure places are mostly shapes, so that category is large.
- **OpenStreetMap coverage is uneven.** Some cities and countries are mapped much more completely than others. The network could end up learning "how well mapped is this city" instead of "how is this city shaped". We need to watch for this, and adding other sources (population, night lights, building footprints from Open Buildings) would help.
- **Boundaries must be comparable.** The default boundary comes from a place-name search, which defines "the city" differently in different countries. For a fair comparison we want the same definition everywhere, such as GHSL Urban Centres or Functional Urban Areas. If a polygon file is placed at `data/boundaries/<slug>.gpkg`, the script uses it instead.
- If a boundary cannot be cleaned and the script falls back to a bounding box, it is flagged in `extraction_summary.csv`. Those cities should not be mixed with the rest without a second look.

## Open questions

- **Unit of analysis:** whole cities, or tiles (neighbourhood-sized patches) inside cities? Tiles give far more training examples. Whole cities match the monocentric/polycentric question more directly.
- **How many cities** do we need? An autoencoder needs dozens to hundreds of cities, or many tiles from fewer cities.
- **Which boundary definition** do we standardise on?
- **Which extra data layers** are worth adding?

## Related work

We are not the first to apply deep learning to urban form, so we should read these before claiming anything is new:

- Moosavi (2017), [Urban morphology meets deep learning](https://arxiv.org/abs/1709.02939): an unsupervised convolutional autoencoder on OpenStreetMap street networks for over a million places worldwide. The closest template to our approach. It uses street networks only.
- [Graph convolutional networks for street network analysis with a case study of urban polycentricity in Chinese cities](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2321229) (not yet read).
- [Classification of Urban Morphology with Deep Learning: Application on Urban Vitality](https://ual.sg/publication/2021-ceus-dl-morphology/) (not yet read).

Where we may add something: combining POIs, buildings and roads in one input, comparing cities across countries with a consistent boundary rule, and checking the learned clusters against classical monocentric/polycentric measures.

## Next steps

1. Run the extraction for more cities and check the summary table.
2. Write the rasterization script (Python).
3. Read the related work above and decide the unit of analysis.
4. Train a first autoencoder on the rasterized cities or tiles.
