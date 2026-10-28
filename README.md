# Spatial Analysis of Cities

A project that classifies cities by their spatial structure: **monocentric** (one dominant centre), **polycentric** (several centres) or **unclear**. Rule-based centre tests on satellite and OpenStreetMap data set each city's type. An unsupervised neural network, trained without labels, is used as an independent check on those types.

The first full run covers 979 urban centres in India (see [Results](#3-results-india-979-cities)).

## Contents

1. [Overview and status](#1-overview-and-status)
2. [Methodology](#2-methodology)
3. [Results: India (979 cities)](#3-results-india-979-cities)
4. [Pilot: Bengaluru](#4-pilot-bengaluru-single-city)
5. [Repository layout](#5-repository-layout)
6. [Setup and running](#6-setup-and-running)
7. [Notes on the data](#7-notes-on-the-data)
8. [Open questions](#8-open-questions)
9. [Related work](#9-related-work)
10. [Next steps](#10-next-steps)

## 1. Overview and status

We download open map data for many cities, turn each city into a set of image-like grids (where the buildings, shops, offices and roads are), and run two things on those grids:

- **Classical centre tests** (rule-based, no learning) decide each city's type and confidence.
- **A neural network** learns, without labels, which cities and neighbourhoods look alike. Its groups are then compared with the classical types.

The type of a city comes from the classical tests, not from the network. The first goal is not a final answer. It is to narrow down broad trends so we know where to look closer.

| Step | State |
|---|---|
| 1. Download city data | Run for India: city list and boundaries (GHS-UCDB), OpenStreetMap (Geofabrik), GHSL built-up (Earth Engine) |
| 2. Consistent city boundaries | Done for country runs (GHSL urban-centre boundaries). One-off city runs still use a place-name search |
| 3. Turn cities into grids | Built (`analysis/rasterize.py`), tested on fake cities, run on India |
| 4. Unsupervised neural network | Built (`analysis/train.py`), tested on fake cities, run on India in tile and city mode |
| 5. Clustering and checking against known measures | Built (`analysis/cluster.py`, `classical.py`, `compare.py`, `classify.py`), tested on fake cities, run on India |

The steps are tested on fake cities with a known answer (`python run_analysis.py --test`, `python run_country.py --test`). There is no independent real-world ground truth for the city types in this repository.

## 2. Methodology

### 2.1 Approach in one picture

```
GHSL urban centres  -> city list and one boundary per city
OpenStreetMap       -> POIs, buildings, roads        \
GHSL built-up (2020) -> built-up share per 100 m cell  > 100 m grids per city
                                                       |
                  +------------------------------------+---------------------+
                  |                                                          |
   Classical centre tests (twice: on GHSL, on OSM)            Autoencoder -> embeddings -> groups
                  |                                                          |
   Final type + confidence  <---------- compared with (ARI, rank correlation, mapping-quality test)
```

The two branches are kept separate on purpose. The network never sees the types, and the types never depend on the network.

### 2.2 Data

| Data | Source | Used for |
|---|---|---|
| City list and boundaries | GHSL Urban Centre Database R2024A (GHS-UCDB, European Commission JRC). Centres with more than 100,000 people | One boundary rule for every city. Population and area |
| Built-up surface | GHSL `GHS_BUILT_S` (Earth Engine asset `JRC/GHSL/P2023A/GHS_BUILT_S`), epoch 2020, 100 m | Primary evidence for the type. Satellite-based, so coverage is equal across cities. Epoch 2020 is the latest built from observations; the 2025 and 2030 layers are projections |
| Points of interest, buildings, roads | OpenStreetMap, from Geofabrik regional files (six India zone files) | Supporting evidence for the type. Optional network input |

GHSL built-up is one year (2020). OpenStreetMap is the map as of the download date. The two are not from the same moment.

### 2.3 City grids

Each city is turned into a stack of grids ("channels"):

- Cells are 100 m squares in the city's own local UTM projection, so a cell is the same size in every city.
- Channels: 10 POI categories (count per cell), building footprint share, major road length, minor road length, and GHSL built-up share.
- The city boundary is used as a mask. It is not shown to the network, so the network cannot learn the shape of administrative borders.
- The city centre is the busiest point, the peak of POI density.

### 2.4 Classical centre tests

These tests use no neural network. They run on a chosen activity layer (`--measure`). The type rule uses two layers, in this order of importance:

1. **GHSL built-up** (`--measure built_up`): the main evidence.
2. **OSM POIs** (`--measure pois`): supporting evidence, used only where the city is reasonably well mapped.

POI density and built-up area stand in for employment, which the classical literature uses but open data does not have. For each layer the city grid is merged into 500 m cells (`--vcell`). The CBD is the busiest point of the layer: for POIs, the centre found when making the grid; for any other layer, that layer's own peak, smoothed over about 1 km. Three tests follow.

| Test | What it does | Default cut-offs |
|---|---|---|
| **Density gradient** | Fits `log(density) = a - b * distance` from the CBD. A steep slope and high R2 mean activity falls away smoothly from one centre. A low R2 means the pattern is lumpier than one centre explains | none. Reported as evidence in `city_types.csv`; it does not set the type |
| **Giuliano-Small style centres** | Cells above a density percentile are joined into contiguous groups (cells touching at an edge or corner count as joined). A group is a centre if it holds enough of the city's activity. The group at the CBD is the main centre, the rest are subcentres | 90th percentile (`--density-pct 90`); at least 3% of the city's activity (`--min-share 0.03`) |
| **McMillen style subcentres** | Fits a smooth curve (LOWESS) of log density against distance from the CBD. Cells far above the curve are candidates. Contiguous candidate groups are subcentres if they hold enough activity, are far enough from the CBD and contain a local density peak | residual above 1.96 (`--z`); at least 3% of activity; at least 3 km from the CBD (`--min-dist-km 3`); LOWESS smoothness 0.3; peak = highest point within 1.5 km after 1 km smoothing |

A test calls a city **monocentric** if it finds no subcentres and **polycentric** if it finds at least one. For a given layer, the city's classical type is the common answer of the two subcentre tests. If they disagree, the type for that layer is **unclear**.

Differences from the literature, which are choices of this project:

- Giuliano and Small (1991) used fixed cut-offs (10 jobs per acre, 10,000 jobs). Here the cut-offs are relative to each city, because POI counts are not jobs and vary with mapping quality.
- McMillen (2001) uses a locally weighted regression over both coordinates. Here a one-dimensional curve against distance is used, with the peak rule to stop the two ends of one stretched-out centre from being counted as subcentres.

The exact rules are written at the top of `analysis/classical.py`.

### 2.5 Final type and confidence

`analysis/classify.py` combines the two layers. **OSM mapping quality** is first worked out for each city as POIs per km² of built-up area, divided by the median city's value:

| `osm_quality` | Ratio to the median city |
|---|---|
| good | 0.5 or more |
| sparse | 0.15 to 0.5 |
| very sparse | below 0.15 |

Then the rule, in order:

| Situation | `final_type` | `confidence` |
|---|---|---|
| GHSL tests agree, OSM tests agree, city is well mapped | the GHSL type | high |
| GHSL tests agree, anything else about OSM (disagree, unclear, or not well mapped) | the GHSL type | medium |
| GHSL tests disagree, city is not very sparsely mapped, and the OSM tests give a clear type | the OSM type | low |
| GHSL tests disagree and OSM cannot settle it | unclear | none |
| No GHSL result | the OSM type if usable (low), otherwise "no data" | low or none |

`city_types.csv` keeps the evidence for every city: `ghsl_type`, `osm_type`, subcentre counts, gradient fits, `osm_quality`, and a plain-words `reason`.

### 2.6 The neural-network check

The network is a check on the classical types. It is not used to set them.

- **Model:** a small convolutional autoencoder. The encoder squeezes an image (a stack of city layers) into an embedding of 16 numbers (`--latent`). The decoder tries to rebuild the image from those numbers. No labels are used.
- **Rotations and flips:** north should not matter. Training shows each image in a random one of its 8 rotations or mirror images, and the final embedding is the average over all 8.
- **Input layers:** by default the network sees only GHSL built-up and roads (`--network-layers ghsl`). In the fake-city test, a network trained on every layer grouped badly mapped cities together instead of grouping by shape. With `--network-layers all` it sees every layer, and the report then checks whether the groups follow mapping quality.
- **Normalisation:** each layer is divided by that city's own average before taking logs (`--normalise city-share`), so the network sees "how much busier than this city's average is this cell", not raw counts.
- **Two ways to cut the examples** (`--mode`):
  - `tile`: many patches per city, 3.2 km squares (32 cells of 100 m). Each city is then described by its mix of patch types.
  - `city`: one image per city, a fixed window of about 40 km centred on the busiest point, 64 x 64 pixels.
- **Grouping:** k-means. In tile mode the default is 8 patch types (a fewer number tends to give only "inner" and "outer" city, too coarse to tell one centre from several), then cities are grouped by their mix of patch types, with the number of city groups chosen by silhouette score. In city mode the number of groups is chosen by silhouette score. Groups are drawn on a 2D map with UMAP, and a profile table shows what each group contains.
- **Comparison with the types** (`analysis/compare.py`, `analysis/classify.py`):
  - a cross-table of network group against final type;
  - the adjusted Rand index (ARI), where 0 means no better than chance and 1 means identical groupings;
  - the rank correlation between the network's main score and the number of GHSL subcentres;
  - a Kruskal-Wallis test of whether the network's groups follow OSM mapping quality. If they do, the groups are partly sorting cities by mapping effort, not by shape.

### 2.7 Safeguards against uneven OSM coverage

OpenStreetMap is mapped much more completely in some places than others, so the network could learn "how well mapped is this city" instead of "how is this city shaped". The safeguards are:

- Layers are divided by the city's own average (`--normalise city-share`).
- The type is set from GHSL first, because satellites cover all cities equally. OSM only supports it.
- The network sees GHSL built-up and roads by default, not POIs.
- Every city gets an `osm_quality` label, and the report tests whether network groups follow it.
- The same boundary rule (GHS-UCDB urban centres) is used for every city. A place-name search defines "the city" differently in different countries.

### 2.8 Limitations of the method

- **Built-up area and POIs are not jobs.** The cut-offs are relative (top 10% of cells, at least 3% of activity). They are choices, not measured values, and results depend on them.
- **GHSL is from one year (2020), OSM is today's map.** The two layers are not from the same moment.
- **The GHSL tests rarely call a city polycentric.** In the India run they find polycentric in 1 of 979 cities. Almost all polycentric calls (39 of 40) come from the OSM tests, after the GHSL tests disagreed. Those calls are low confidence.
- **Giant urban centres.** The GHSL database joins built-up areas that touch on its 1 km grid, so parts of Kerala and the Gangetic plain can come out as one "city" the size of a district. They are flagged `large_area=True`. Read their type as a statement about the region.
- **The network's groups do not line up with the classical types** (see Results), and in tile mode they partly follow mapping quality. The network check therefore does not confirm the classical types.
- **No real-world ground truth.** Types are only checked against known answers on fake cities.
- **Buildings are counted at their centre point**, so a very large building adds all its area to one 100 m cell.

## 3. Results: India (979 cities)

All numbers below are from `data/india/analysis/classification/report.md` and `city_types.csv`. Read the report for the full tables.

| `final_type` | Cities | High | Medium | Low | No confidence |
|---|---|---|---|---|---|
| monocentric | 802 | 404 | 308 | 90 | 0 |
| polycentric | 40 | 1 | 0 | 39 | 0 |
| unclear | 137 | 0 | 0 | 0 | 137 |

| Size class | monocentric | polycentric | unclear |
|---|---|---|---|
| 0.1-0.5 million | 681 | 16 | 114 |
| 0.5-1 million | 69 | 9 | 13 |
| 1-5 million | 41 | 13 | 10 |
| 5 million+ | 11 | 2 | 0 |

OSM mapping quality: 711 cities good, 200 sparse, 68 very sparse.

What to take from it, and what not to:

- **Most cities come out monocentric, and most polycentric calls are weak.** 39 of the 40 polycentric calls are low confidence. Treat the polycentric list as candidates to check, not as established results.
- **GHSL and OSM agree mostly on monocentric.** Where both give a clear type (507 cities) they agree for 93%. Almost all of that is 473 cities that are monocentric in both. Only 1 city is polycentric in both.
- **The network's groups barely match the classical types.**
  - Tile mode: ARI -0.01 (chance). These groups also follow OSM mapping quality (Kruskal-Wallis p = 6.67e-27), so they partly sort cities by mapping effort, not by shape. Tile-mode groups exist for 933 of the 979 cities.
  - City mode: ARI 0.11, slightly above chance but still low. The report's mapping-quality check does not flag these groups.
- **OSM tests give no type for 164 cities** (blank `osm_type`). For those, the type rests on GHSL alone.
- **The largest "polycentric" city is a joined area.** Hajipur (about 9.8 million people, 3,153 km²) is the only city flagged `large_area=True`. Check it by hand (`report.md`, "Very large urban centres").

Outputs are in `data/india/analysis/classification/`: `report.md`, `city_types.csv`, `city_types.gpkg` (with boundaries, for QGIS) and `city_types_map.png`.

## 4. Pilot: Bengaluru (single city)

The first real run used one city, Bengaluru, in tile mode, before the country run. It is kept here as a worked example. With one city, the network's patch types describe areas within Bengaluru and are not comparable across cities.

OSM extraction (October 2026): 54,186 POIs (40,855 mapped as points, 13,331 as shapes), 736,627 buildings, 108,713 road lines, boundary from a place-name search.

- **The POI tests disagreed.** The Giuliano-Small style test found one subcentre, south of the main centre. The McMillen style test found none. So the OSM result was "unclear". These tests count POIs, which likely under-count job hubs such as tech parks. Rerunning with `--measure poi_offices` is a pending check.
- **In the India run Bengaluru is monocentric, with medium confidence.** The GHSL tests agree on monocentric, and the OSM tests are still unclear (see `reason` in `city_types.csv`).
- **The network's 8 patch types formed rings:** a commercial core, a dense residential ring, mixed suburbs, outer residential, and the city edge. Rings are expected in any city, polycentric ones included, so they do not by themselves show monocentricity.
- **One exception:** an office-and-leisure patch type on the east side, away from the core. It may be a second, office-type centre that the POI-based tests miss. It still needs checking on a map (`data/rasters/bengaluru.tif` in QGIS).

## 5. Repository layout

```
setup_env.py                One-command setup: venv, Python packages, R packages
data_extraction/
  data_extraction_osm.R     Step 1: boundary, POIs, buildings, roads from OSM (one-off cities)
  amenity_wrap.R            Groups raw OSM tags into POI categories and plots them
  ghsl_builtup_capitals.R   Step 1: GHSL built-up surface (10 m) from Google Earth Engine (older version)
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
site/                       Scroll-driven results site (Vite + TypeScript)
data/                       Inputs and outputs (large files not committed, see below)
```

## 6. Setup and running

### 6.1 Setup

`python setup_env.py` creates a virtual environment and installs the Python and R packages (`--skip-r` or `--skip-python` for one half, `--dry-run` to preview). Run `python setup_env.py --help` for the other options. Python 3.10 or newer is needed.

Check everything works first. This makes 16 fake cities (half with one centre, half with several), runs every step on them, and checks the answers. No real data or internet is needed, and it takes a few minutes:

```
python run_analysis.py --test
python run_country.py --test
```

### 6.2 Country run (India)

`run_country.py` runs the whole chain for every GHSL urban centre in a country with more than 100,000 people. It uses one boundary rule for all cities, pulls OSM and GHSL data for each, runs steps 2-5, and gives every city a type.

```
python run_country.py --country India --project MY_GCP_PROJECT --limit 20     the 20 largest cities, as a first check
python run_country.py --country India --project MY_GCP_PROJECT                then all of them
```

`MY_GCP_PROJECT` is your Google Cloud project for Earth Engine (or set the environment variable `EE_PROJECT`). To rerun only the analysis and classification, for example with Spark:

```
python run_country.py --country India --project MY_GCP_PROJECT --only analysis classify --engine spark --force
```

Stages, each resumable (rerun the same command after a crash and finished work is kept):

| Stage | Script | What it does |
|---|---|---|
| `list` | `data_extraction/build_city_list.py` | GHSL urban centres over `--min-pop` (default 100,000), one boundary each. Downloads the global UCDB once (about 1.7 GB, kept in `data/uc/`) |
| `osm` | `data_extraction/osm_batch_extract.py` | Downloads the six Geofabrik India zone files (about 1.6 GB, resumable, checksum-checked; kept in `data/osm_pbf/`), then reads each file once and hands every POI, building and road to the cities it lies in |
| `ghsl` | `data_extraction/ghsl_builtup_100m.py` | GHSL built-up surface at 100 m from Earth Engine, one request per city (needs `--project`) |
| `analysis` | steps 2-5 | Grids; classical tests on GHSL built-up and on OSM POIs; autoencoder and groups in tile and city mode |
| `classify` | `analysis/classify.py` | One type per city, a map and a report |

Everything goes into `data/india/`, so it does not mix with other runs or the earlier Bengaluru run.

**OSM zone files.** If you prefer to download them yourself, put these into `data/osm_pbf/`, keeping the file names as they are (the script finds them there and does not download them again):

```
https://download.geofabrik.de/asia/india/central-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/eastern-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/northern-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/western-zone-latest.osm.pbf
https://download.geofabrik.de/asia/india/north-eastern-zone-latest.osm.pbf
```

**Things to know before a full run:**

- Space: about 1.7 GB (UCDB) + 1.6 GB (OSM) + several GB of temporary files while reading OSM (in `data/osm_pbf/tmp/`), plus the city files. Keep the project outside OneDrive.
- Time: the OSM stage reads each zone file three times (points, areas, lines). Expect hours for all of India, but it does not grow with the number of cities. Steps 2-5 take roughly a few seconds to a minute per city. `--workers 4` grids several cities at once if you have the memory.
- `osm_batch_extract.py` replaces the R extraction for country runs. The R script still works for one-off cities.

### 6.3 City lists for other countries

`build_city_list.py` makes the list of GHSL urban centres and one boundary per city. Its defaults are `--country "United States"` and `--min-pop 50000`, so pass `--country` explicitly (`run_country.py` does this for you, with `--min-pop 100000`):

```
python data_extraction/build_city_list.py --country India --min-pop 100000
python data_extraction/ghsl_builtup_100m.py --project=MY_GCP_PROJECT --cities-csv data/india/uc/india_cities.csv --data-dir data/india
```

`build_city_list.py` also takes `--ucdb` (a file you already downloaded). `ghsl_builtup_100m.py` also takes `--limit`, `--only <slug ...>` and `--epoch` (default 2020), and its default city list is `data/uc/us_cities.csv`. Both scripts document their output at the top of the file. `ghsl_builtup_capitals.R` is the older 10 m version used for the Indian cities.

### 6.4 One-off cities (R extraction, then steps 2-5)

Install the R packages once: `osmextract`, `osmdata`, `sf`, `dplyr`, `ggplot2`, and optionally `here`. From the project root:

```
Rscript data_extraction/data_extraction_osm.R "Bengaluru, India"
Rscript data_extraction/data_extraction_osm.R "Pune, India" "Delhi, India"
Rscript data_extraction/amenity_wrap.R bengaluru
```

Useful flags: `--force` redoes cities that already have output files; `--strict` stops instead of warning if a boundary falls back to a rough box. For each city (`<slug>` is the lower-case city name) you get `<slug>_boundary.gpkg`, `<slug>_pois.gpkg`, `<slug>_buildings.gpkg`, `<slug>_roads.gpkg`, plus `extraction_summary.csv` (status, counts, boundary source and area in km² per city) and `osm_pbf/` (cached OpenStreetMap downloads from Geofabrik, e.g. https://download.geofabrik.de/asia/india.html). Output files are large and are not committed to git. Everyone generates them locally.

Then the Python steps:

```
python run_analysis.py                      both modes, every city in data/
python run_analysis.py --mode tile          only patches inside cities
python run_analysis.py --mode city bengaluru pune delhi
```

Tile mode works with a handful of cities. City mode needs dozens to hundreds before a network can learn much. Each step can also be run on its own, for example `python -m analysis.cluster --mode tile --k 5`. Add `--help` to any command to see its options. Useful ones:

- `--cell 200` bigger grid cells (smoother, faster)
- `--normalise absolute` compare raw counts instead of each city relative to its own average
- `--channels poi_offices,poi_retail,buildings` choose which layers the network sees
- `--measure poi_offices` use offices instead of all POIs for the classical tests
- `--skip-rasterize`, `--skip-train`, `--skip-classical` reuse earlier results

Where the results go:

- `data/rasters/`: one grid per city (`.npz`, a `.tif` you can open in QGIS, and a preview image)
- `data/analysis/classical/`: classical measures per city and a map of each city's centres
- `data/analysis/city/` and `data/analysis/tile/`: embeddings, groups, maps, and `comparison_report.md`, the plain-language summary of whether the network's groups match the classical tests

### 6.5 Using Spark (`--engine spark`)

Two steps can run on Apache Spark instead of plain Python. The results are the same. Spark does the work in parallel and would scale to many more cities, or to a cluster:

- **Step 2 (Spark SQL):** every city's POIs, buildings and road pieces become one table, about 1 million rows per city, saved as Parquet and partitioned by city. Spark works out the grid cell of every row and sums them by city, layer and cell, all cities in one job (`aggregate_spark` in `analysis/rasterize.py`).
- **Step 4 (Spark MLlib):** k-means clustering with `pyspark.ml.clustering.KMeans`, with the number of groups chosen by Spark's `ClusteringEvaluator` silhouette (`spark_group` in `analysis/cluster.py`).

Spark runs in local mode (one worker per CPU core). Only Spark SQL and MLlib are used, no Python functions inside Spark, which avoids the usual Windows problems.

Setup (once):

1. Install Java 17 or newer. Windows: the "Temurin 21 JDK" from https://adoptium.net, and tick "Set JAVA_HOME variable" in the installer. Restart VS Code, then check with `java -version`.
2. `pip install pyspark pyarrow` (already in `requirements.txt`).
3. Check it works: `python run_analysis.py --test-spark`. This rasterizes 6 fake cities with and without Spark and checks the grids are identical, then checks Spark's k-means.

Run: `python run_analysis.py --mode tile --engine spark`, or one step at a time with `python -m analysis.rasterize --engine spark --force` and `python -m analysis.cluster --mode tile --method spark-kmeans`.

If Spark on Windows stops with an error mentioning `winutils.exe` or `HADOOP_HOME`, download `winutils.exe` and `hadoop.dll` for Hadoop 3.3 into a folder such as `C:\hadoop\bin`. Then set `HADOOP_HOME` to `C:\hadoop` and add `C:\hadoop\bin` to `Path`. This was tested on Linux; the steps above are what Spark needs on Windows.

### 6.6 Results site

The `site/` folder is a scroll-driven results page (Vite + TypeScript). It reads `site/public/city_types.csv`, a copy of `data/india/analysis/classification/city_types.csv`. Copy the file again after any rerun.

```
cd site
npm install
npm run dev        local preview
npm run build      type-check and production build
```

## 7. Notes on the data

- **POIs come from two places.** Some are single points on the map (`poi_source = "node"`), and some are drawn as shapes and converted to one point each (`poi_source = "area"`). Some places may appear twice, once as each. Parks and other leisure places are mostly shapes, so that category is large.
- **OpenStreetMap coverage is uneven.** See section 2.7 for how this is handled. Adding other sources (population, night lights, building footprints from Open Buildings) would help further.
- **Boundaries.** Country runs use GHS-UCDB urban-centre boundaries. For one-off cities the default boundary comes from a place-name search. If a polygon file is placed at `data/boundaries/<slug>.gpkg`, the script uses it instead. If a boundary cannot be cleaned and the script falls back to a bounding box, it is flagged in `extraction_summary.csv`. Those cities should not be mixed with the rest without a second look.
- **Only small result files in `data/` are tracked in git.** Everything else, including the large downloads, is ignored (see `.gitignore`).

## 8. Open questions

- **Unit of analysis:** both are built (`--mode city` and `--mode tile`). Which one is more informative on real cities? On the India run, city mode matches the types slightly better (ARI 0.11 against -0.01), and tile mode follows mapping quality.
- **Which extra data layers** are worth adding (population, night lights, Open Buildings)? A new layer is added in `analysis/rasterize.py` and `analysis/common.py` (`CHANNELS`).

## 9. Related work

We are not the first to apply deep learning to urban form, so we should read these before claiming anything is new:

- Moosavi (2017), [Urban morphology meets deep learning](https://arxiv.org/abs/1709.02939): an unsupervised convolutional autoencoder on OpenStreetMap street networks for over a million places worldwide. The closest template to our approach. It uses street networks only.
- [Graph convolutional networks for street network analysis with a case study of urban polycentricity in Chinese cities](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2321229) (not yet read).
- [Classification of Urban Morphology with Deep Learning: Application on Urban Vitality](https://ual.sg/publication/2021-ceus-dl-morphology/) (not yet read).

Where we may add something: combining POIs, buildings and roads in one input, comparing cities across countries with a consistent boundary rule, and checking the learned clusters against classical monocentric and polycentric measures.

## 10. Next steps

1. Look at `cluster_profiles.png` and `tile_maps.png` to name the network's groups, and at the very large urban centres by hand.
2. Read the related work above.
