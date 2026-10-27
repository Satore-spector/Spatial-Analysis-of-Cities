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

A country-wide run (every GHSL urban centre over 100,000 people, e.g. all of India) is built and tested offline: see "Classifying every city in a country" below. It has not yet been run on real data.

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
setup_env.py                One-command setup: venv, Python packages, R packages
data_extraction/
  data_extraction_osm.R     Step 1: boundary, POIs, buildings, roads from OSM
  amenity_wrap.R            Groups raw OSM tags into POI categories and plots them
  ghsl_builtup_capitals.R   Step 1: GHSL built-up surface (10 m) from Google Earth Engine
  build_city_list.py        Step 1: list of urban centres + boundaries from GHS-UCDB (any country)
  osm_batch_extract.py      Step 1: OSM for hundreds of cities, one read per regional file
  ghsl_builtup_100m.py      Step 1: GHSL built-up surface (100 m) for a city list
run_pipeline.R              Runs the R steps in order (with --check / --test modes)
analysis/                   Steps 2-5 in Python
  rasterize.py              Step 2: .gpkg files -> grids (numpy or Spark SQL)
  spark_utils.py            Starts Spark for --engine spark
  samples.py                Step 3: grids -> training examples (city or tile mode)
  model.py, train.py        Step 3: the autoencoder and its training
  cluster.py                Step 4: groups, 2D map, group profiles
  classical.py              Step 5: rule-based subcentre tests
  compare.py                Step 5: do the network's groups match the tests?
  classify.py               Country runs: one type per city, map, report
run_analysis.py             Runs steps 2-5 in one go
run_country.py              Classifies every city in a country (list -> OSM -> GHSL -> 2-5 -> types)
requirements.txt            Python packages
tests/                      Offline tests (fake data, no downloads)
data/                       Inputs and outputs (not committed, see below)
```

## Setup

`python setup_env.py` creates a virtual environment and installs the Python and R packages (`--skip-r` or `--skip-python` for one half, `--dry-run` to preview). Run `python setup_env.py --help` for the other options.

## How to run the extraction

Install the R packages once: `osmextract`, `osmdata`, `sf`, `dplyr`, `ggplot2`, and optionally `here`.

From the project root:

```
Rscript data_extraction/data_extraction_osm.R "Bengaluru, India"
Rscript data_extraction/data_extraction_osm.R "Pune, India" "Delhi, India"
Rscript data_extraction/amenity_wrap.R bengaluru
```

Useful flags for the extraction script:

- `--force` redo cities that already have output files
- `--strict` stop instead of warning if a boundary falls back to a rough box

What you get in `data/` for each city (`<slug>` is the lower-case city name):

- `<slug>_boundary.gpkg`, `<slug>_pois.gpkg`, `<slug>_buildings.gpkg`, `<slug>_roads.gpkg`
- `extraction_summary.csv`, one row per city with status, counts, boundary source and area in km²
- `osm_pbf/`, cached OpenStreetMap downloads so we do not download the same region twice (regional extracts come from Geofabrik, e.g. https://download.geofabrik.de/asia/india.html)

Output files are large and are not committed to git. Everyone generates them locally.

### US cities (GHSL urban centres)

For a consistent boundary definition across many cities, the US list comes from the GHSL Urban Centre Database instead of a place-name search. From the project root:

```
python data_extraction/build_city_list.py                    # writes data/uc/us_cities.csv and one boundary per city
python data_extraction/ghsl_builtup_100m.py --project=MY_GCP_PROJECT
```

`build_city_list.py` takes `--min-pop`, `--country` and `--ucdb`; `ghsl_builtup_100m.py` takes `--limit` and `--only <slug ...>`. Both scripts document their output at the top of the file. `ghsl_builtup_capitals.R` is the older 10 m version used for the Indian cities.

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

## Classifying every city in a country (India)

`run_country.py` runs the whole chain for every GHSL urban centre in a country with more than 100,000 people. It uses one boundary rule for all cities, pulls OSM and GHSL data for each, runs steps 2-5, and gives every city a type.

```
python run_country.py --test                                  offline test on fake cities (3-5 min, no internet)
python run_country.py --country India --project MY_GCP_PROJECT --limit 20     first, the 20 largest cities
python run_country.py --country India --project MY_GCP_PROJECT                then all of them
```

Stages, each resumable (rerun the same command after a crash and finished work is kept):

| Stage | Script | What it does |
|---|---|---|
| `list` | `data_extraction/build_city_list.py` | GHSL urban centres over `--min-pop` (default 100,000), one boundary each. Downloads the global UCDB once (about 1.7 GB, kept in `data/uc/`) |
| `osm` | `data_extraction/osm_batch_extract.py` | Downloads the six Geofabrik India zone files (about 1.6 GB, resumable, checksum-checked; kept in `data/osm_pbf/`), then reads each file once and hands every POI, building and road to the cities it lies in |
| `ghsl` | `data_extraction/ghsl_builtup_100m.py` | GHSL built-up surface at 100 m from Earth Engine, one request per city (needs `--project`) |
| `analysis` | steps 2-5 | Grids; classical tests on GHSL built-up and on OSM POIs; autoencoder and groups in tile and city mode |
| `classify` | `analysis/classify.py` | One type per city, a map and a report |

Everything goes into `data/india/`, so it does not mix with the US runs or the earlier Bengaluru run. Results are in `data/india/analysis/classification/`:

- `report.md`: the answer in plain words (counts of each type, by city size, GHSL vs OSM agreement, network check, largest polycentric cities, caveats)
- `city_types.csv`: one row per city with its type, confidence and all the evidence
- `city_types.gpkg`: the same with boundaries, to open in QGIS
- `city_types_map.png`: every city on a map, by type and by network group

**How a city's type is decided (GHSL first, OSM as support).** The classical centre tests run on the GHSL built-up layer, which comes from satellites and covers every city equally. If the Giuliano-Small and McMillen style tests agree, that is the type: confidence "high" if the OSM POI tests agree and the city is well mapped, otherwise "medium". If the GHSL tests disagree, the OSM tests decide (confidence "low"), but only for cities that are not very sparsely mapped; otherwise the city is "unclear". Each city's OSM mapping quality is POIs per km² of built-up area, compared with the median city.

**What the network learns from.** By default the network sees only GHSL built-up and roads (`--network-layers ghsl`), not POIs or buildings. In the fake-city test, a network trained on every layer grouped the badly mapped cities together instead of grouping by shape. With built-up and roads it separated one-centre from several-centre cities. `--network-layers all` uses every layer; the report then checks whether the groups follow mapping quality and warns if they do.

**Things to know before the full run:**
- Space: about 1.7 GB (UCDB) + 1.6 GB (OSM) + several GB of temporary files while reading OSM (in `data/osm_pbf/tmp/`), plus the city files. Keep the project outside OneDrive.
- Time: the OSM stage reads each zone file three times (points, areas, lines); expect hours for all of India, but it does not grow with the number of cities. Steps 2-5 take roughly a few seconds to a minute per city; `--workers 4` grids several cities at once if you have the memory.
- Very large urban centres: the GHSL database joins built-up areas that touch on its 1 km grid, so parts of Kerala and the Gangetic plain can come out as one "city" the size of a district. The list marks them `large_area=True` and the report lists them; read their type as a statement about the region.
- With hundreds of cities, city mode (one image per city) has enough examples to be worth reading, alongside tile mode.
- `osm_batch_extract.py` replaces the R extraction for country runs; the R script still works for one-off cities.
- Tested offline only (fake cities, fake OSM files, fake GHSL rasters). The first real run with `--limit 20` is the check that the downloads and Earth Engine work on your machine.

## First three-city run (US, one boundary definition)

A small, controlled test before scaling up. Three US cities from the literature, all with the same GHSL urban-centre boundary and similar OpenStreetMap quality, so differences are less likely to come from mapping effort or boundary rules:

| Role | City | Why |
|---|---|---|
| Monocentric | Boston | One dominant downtown |
| Polycentric | Los Angeles | The standard example of many centres |
| Other | Atlanta | Low-density, dispersed, no strong core |

These labels come from the literature and are expectations, not facts: the classical tests (`classical.py`) decide whether the data agrees. The network never sees the labels; it is unsupervised. The labels are only used afterwards to check whether the groups it finds line up with them. Three cities is the minimum for a comparison and is not enough to claim anything general.

```
python data_extraction/build_city_list.py
grep -iE "^(bostonma|losangelesca|atlantaga)|,Boston|,Los Angeles|,Atlanta" data/uc/us_cities.csv   # confirm slugs and city strings
Rscript data_extraction/data_extraction_osm.R "Boston MA, USA" "Los Angeles CA, USA" "Atlanta GA, USA"
python data_extraction/ghsl_builtup_100m.py --project=MY_GCP_PROJECT --only bostonma losangelesca atlantaga
python run_analysis.py --mode tile bostonma losangelesca atlantaga
```

The city string passed to R must produce the same slug as in `us_cities.csv` (letters only, e.g. `bostonma`). Los Angeles and Atlanta are big: expect large downloads (Geofabrik California and Georgia extracts) and long runs, and add `--cell 200` if memory runs out.

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

1. `python run_country.py --test`, then `python run_country.py --country India --project MY_GCP_PROJECT --limit 20` as a first real run; check `osm_extraction_summary.csv`, `ghsl/ghsl_100m_summary.csv` and the report.
2. Then all of India. Read `data/india/analysis/classification/report.md`, especially the GHSL vs OSM agreement and the mapping-quality check.
3. Look at `cluster_profiles.png` and `tile_maps.png` to name the network's groups, and at the very large urban centres by hand.
4. Read the related work above.

## OSM DATA
Download these into the project’s data\osm_pbf\ folder, keeping the file names as they are (the script finds them there and won’t download again):

https://download.geofabrik.de/asia/india/central-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/eastern-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/northern-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/western-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/north-eastern-zone-latest.osm.pbf 

command to run:
python run_country.py --country India --project bdacer --limit 20 --only analysis classify --engine spark --force
--to check if it works on 20 largest cities
python run_country.py --country India --project bdacer --only analysis classify --engine spark --force
