# Spatial Analysis of Cities

A project to find out whether cities can be sorted by their spatial structure (for example **monocentric**: one dominant centre, or **polycentric**: several centres) using patterns learned by a neural network, rather than rules we decide in advance.

This README explains the plan, what exists today, and where help is needed.

## The idea in one paragraph

We download open map data for many cities, turn each city into a set of image-like grids (where the shops, offices, buildings and roads are), and train a neural network **without labels** to learn what makes cities look similar or different. We then cluster what it learned and check whether the groups match known ideas like monocentric and polycentric. The first goal is not a final answer. It is to narrow down broad trends so we know where to look closer.

## Status

| Step | State |
|---|---|
| 1. Download city data (OpenStreetMap + GHSL built-up) | OSM extraction works (run for Bengaluru). The GHSL pull is built but still being tested on real data |
| 2. Consistent city boundaries | Partly done (see below) |
| 3. Turn cities into grids | Built (`analysis/rasterize.py`); tested on fake cities, run on Bengaluru |
| 4. Unsupervised neural network | Built (`analysis/train.py`); tested on fake cities, run on Bengaluru (tile mode) |
| 5. Clustering and checking against known measures | Built (`analysis/cluster.py`, `classical.py`, `compare.py`); tested on fake cities, run on Bengaluru |

Steps 3-5 are tested on fake cities with a known answer (`python run_analysis.py --test`) and have run on one real city, Bengaluru, in tile mode. Comparing cities with each other needs at least 3 cities; that is the next run.

## First results: Bengaluru (one city, tile mode)

OSM extraction (October 2026): 54,186 POIs (40,855 mapped as points, 13,331 as shapes), 736,627 buildings, 108,713 road lines, boundary from the place-name search.

- **Classical tests disagree.** The Giuliano-Small style test finds one subcentre, south of the main centre. The McMillen style test finds none. So Bengaluru comes out as "unclear". Activity is spread across the city with many medium-busy patches, so nothing stands out sharply enough for the stricter McMillen test. These tests count POIs (shops, amenities), which likely under-count job hubs such as tech parks; rerunning with `--measure poi_offices` is the next check.
- **The network's 8 patch types form rings.** A commercial core (dense in retail, finance and food) sits in the middle, surrounded by a dense residential ring, then mixed suburbs, then outer residential and the city edge. Rings like this are expected in any city, polycentric ones included, because activity falls off with distance from the main centre, so they do not by themselves show that Bengaluru is monocentric. The classical tests above disagree, so for now Bengaluru is "unclear".
- **One exception: an office-and-leisure patch type on the east side,** away from the core. This may be a second, office-type centre that the POI-based classical tests miss. Where exactly it is still needs checking on a map (`data/rasters/bengaluru.tif` in QGIS).

With one city these are descriptions of Bengaluru, not comparisons: a patch type means "this kind of area within Bengaluru". Running several cities together makes the types shared and comparable.

## How the pipeline works

1. **Extract** (R): for each city, save the boundary, points of interest (POIs), buildings and roads as `.gpkg` files, and optionally the GHSL built-up raster.
2. **Rasterize** (Python): turn each city into grids of 100 m cells in its local projection, one layer per thing: 10 POI categories, building footprint share, major and minor road length, and GHSL built-up share. The busiest point (peak of POI density) is marked as the city's centre.
3. **Learn** (Python): train a convolutional autoencoder that squeezes each example into a short list of numbers (an embedding). Rotations and flips count as the same layout, since north should not matter. There are two ways to cut the examples, chosen with `--mode`:
   - `city`: one image per city, a fixed window (about 40 km) centred on the city. Matches the question directly but needs dozens to hundreds of cities.
   - `tile`: many 3.2 km patches per city. Works with a handful of cities; each city is then described by its mix of patch types.
4. **Cluster**: group the embeddings with k-means or HDBSCAN, and draw them on a 2D map with UMAP. In tile mode the default is 8 patch types (fewer tends to give just "inner" and "outer" city, too coarse to tell one centre from several); cities are then grouped by their mix of patch types. In city mode the number of groups is picked by silhouette score. A table shows what each group contains, so groups can be named ("office-heavy core", "residential edge" ...).
5. **Check**: run classical, rule-based tests on the same grids (density-gradient fit, Giuliano-Small style centres, McMillen style subcentres) and measure how well the network's groups match them.

The two languages only meet at the files. R writes `.gpkg` and `.tif`, and Python reads them.

## Repo layout

```
data_extraction/
  data_extraction_osm.R     Step 1: boundary, POIs, buildings, roads from OSM
  amenitiy_wrap.r           Groups raw OSM tags into POI categories and plots them
  ghsl_builtup_capitals.R   Step 1: GHSL built-up surface from Google Earth Engine
run_pipeline.R              Runs the R steps in order (with --check / --test modes)
analysis/                   Steps 2-5 in Python
  rasterize.py              Step 2: .gpkg files -> grids (numpy or Spark SQL)
  spark_utils.py            Starts Spark for --engine spark
  samples.py                Step 3: grids -> training examples (city or tile mode)
  model.py, train.py        Step 3: the autoencoder and its training
  cluster.py                Step 4: groups, 2D map, group profiles
  classical.py              Step 5: rule-based subcentre tests
  compare.py                Step 5: do the network's groups match the tests?
run_analysis.py             Runs steps 2-5 in one go
requirements.txt            Python packages
tests/                      Offline tests (fake data, no downloads)
data/                       Inputs and outputs (not committed, see below)
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
- `osm_pbf/`, cached OpenStreetMap downloads so we do not download the same region twice (regional extracts come from Geofabrik, e.g. https://download.geofabrik.de/asia/india.html)

Output files are large and are not committed to git. Everyone generates them locally.

## How to run steps 2-5 (Python)

Install once (Python 3.10 or newer): `pip install -r requirements.txt`

Check everything works first. This makes 16 fake cities (half with one centre, half with several), runs every step on them, and checks the answers. No real data or internet needed; it takes a few minutes:

```
python run_analysis.py --test
```

Then, after the R extraction has filled `data/`:

```
python run_analysis.py                      both modes, every city in data/
python run_analysis.py --mode tile          only patches inside cities
python run_analysis.py --mode city bengaluru pune delhi
```

Each step can also be run on its own, for example `python -m analysis.cluster --mode tile --k 5`. Add `--help` to any command to see its options. Useful ones:

- `--cell 200` bigger grid cells (smoother, faster)
- `--normalise absolute` compare raw counts instead of each city relative to its own average (see below)
- `--channels poi_offices,poi_retail,buildings` choose which layers the network sees
- `--measure poi_offices` use offices instead of all POIs for the classical tests
- `--skip-rasterize`, `--skip-train`, `--skip-classical` reuse earlier results

### Using Spark (`--engine spark`)

Two steps can run on Apache Spark instead of plain Python. The results are the same; Spark does the work in parallel and would scale to many more cities, or to a cluster:

- **Step 2 (Spark SQL):** every city's POIs, buildings and road pieces become one table, about 1 million rows per city, saved as Parquet and partitioned by city. Spark works out the grid cell of every row and sums them by city, layer and cell, all cities in one job (`aggregate_spark` in `analysis/rasterize.py`).
- **Step 4 (Spark MLlib):** k-means clustering with `pyspark.ml.clustering.KMeans`, with the number of groups chosen by Spark's `ClusteringEvaluator` silhouette (`spark_group` in `analysis/cluster.py`).

Spark runs in local mode (one worker per CPU core). Only Spark SQL and MLlib are used, no Python functions inside Spark, which avoids the usual Windows problems.

Setup (once):

1. Install Java 17 or newer. Windows: the "Temurin 21 JDK" from https://adoptium.net, and tick "Set JAVA_HOME variable" in the installer. Restart VS Code, then check with `java -version`.
2. `pip install pyspark pyarrow` (already in `requirements.txt`).
3. Check it works: `python run_analysis.py --test-spark`. This rasterizes 6 fake cities with and without Spark and checks the grids are identical, then checks Spark's k-means.

Run: `python run_analysis.py --mode tile --engine spark`, or one step at a time with `python -m analysis.rasterize --engine spark --force` and `python -m analysis.cluster --mode tile --method spark-kmeans`.

If Spark on Windows stops with an error mentioning `winutils.exe` or `HADOOP_HOME`, download `winutils.exe` and `hadoop.dll` for Hadoop 3.3 into a folder such as `C:\hadoop\bin`. Then set `HADOOP_HOME` to `C:\hadoop` and add `C:\hadoop\bin` to `Path`. This was tested on Linux; the steps above are what Spark needs on Windows.

Where the results go:

- `data/rasters/` one grid per city (`.npz`, a `.tif` you can open in QGIS, and a preview image)
- `data/analysis/classical/` classical measures per city and a map of each city's centres
- `data/analysis/city/` and `data/analysis/tile/`: embeddings, groups, maps, and `comparison_report.md`, the plain-language summary of whether the network's groups match the classical tests

## Things to know about the data

- **POIs come from two places.** Some are single points on the map (`poi_source = "node"`), and some are drawn as shapes and converted to one point each (`poi_source = "area"`). Some places may appear twice, once as each. Parks and other leisure places are mostly shapes, so that category is large.
- **OpenStreetMap coverage is uneven.** Some cities and countries are mapped much more completely than others. The network could end up learning "how well mapped is this city" instead of "how is this city shaped". We need to watch for this, and adding other sources (population, night lights, building footprints from Open Buildings) would help.
- **Boundaries must be comparable.** The default boundary comes from a place-name search, which defines "the city" differently in different countries. For a fair comparison we want the same definition everywhere, such as GHSL Urban Centres or Functional Urban Areas. If a polygon file is placed at `data/boundaries/<slug>.gpkg`, the script uses it instead.
- If a boundary cannot be cleaned and the script falls back to a bounding box, it is flagged in `extraction_summary.csv`. Those cities should not be mixed with the rest without a second look.
- **Against the mapping-quality problem**, step 3 by default divides each layer by the city's own average (`--normalise city-share`). The network then sees "where is this city busier than its own average", not raw counts, so a well-mapped city does not look different just for having more points. The GHSL built-up layer is also used when every city has it, since satellites cover all cities equally.
- **The boundary itself is not shown to the network.** Otherwise it could learn the shape of administrative borders.
- **The classical tests use POIs, not jobs.** The literature uses employment. Here POI density stands in for it, and cut-offs are relative to each city (top 10% of cells, at least 3% of the city's activity) instead of fixed numbers like 10 jobs per acre. The exact rules are at the top of `analysis/classical.py`.
- **Buildings are counted at their centre point**, so a very large building adds all its area to one 100 m cell. Fine for most cities; worth knowing for big campuses or factories.

## Open questions

- **Unit of analysis:** both are built (`--mode city` / `--mode tile`). Which one gives clearer results on real cities?
- **How many cities** do we need? City mode needs dozens to hundreds. Tile mode works with fewer, but more cities still make the groups more trustworthy.
- **Which boundary definition** do we standardise on? This matters even more now, because the city-mode window is centred on the busiest point and tiles are kept only if half of them is inside the boundary.
- **Which extra data layers** are worth adding (population, night lights, Open Buildings)? A new layer is added in `analysis/rasterize.py` and `analysis/common.py` (`CHANNELS`).

## Related work

We are not the first to apply deep learning to urban form, so we should read these before claiming anything is new:

- Moosavi (2017), [Urban morphology meets deep learning](https://arxiv.org/abs/1709.02939): an unsupervised convolutional autoencoder on OpenStreetMap street networks for over a million places worldwide. The closest template to our approach. It uses street networks only.
- [Graph convolutional networks for street network analysis with a case study of urban polycentricity in Chinese cities](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2321229) (not yet read).
- [Classification of Urban Morphology with Deep Learning: Application on Urban Vitality](https://ual.sg/publication/2021-ceus-dl-morphology/) (not yet read).

Where we may add something: combining POIs, buildings and roads in one input, comparing cities across countries with a consistent boundary rule, and checking the learned clusters against classical monocentric/polycentric measures.

## Next steps

1. Run the extraction for more cities and check the summary table.
2. Run `python run_analysis.py --mode tile` on the real cities and read `comparison_report.md`.
3. Look at `cluster_profiles.png` and `tile_maps.png` to name the groups, and check they are not just "well mapped vs poorly mapped".
4. Read the related work above, and settle the boundary definition before adding many more cities.
