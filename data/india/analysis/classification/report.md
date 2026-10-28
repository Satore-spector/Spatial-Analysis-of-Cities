# City types

979 cities. Type from the classical centre tests on GHSL built-up (satellite), with OpenStreetMap POIs as support. Rules: see analysis/classify.py.

## How many of each

| final_type | cities |
|---|---|
| monocentric | 802 |
| unclear | 137 |
| polycentric | 40 |

| final_type | - | high | low | medium |
|---|---|---|---|---|
| monocentric | 0 | 404 | 90 | 308 |
| polycentric | 0 | 1 | 39 | 0 |
| unclear | 137 | 0 | 0 | 0 |

## By city size

| size_class | monocentric | polycentric | unclear |
|---|---|---|---|
| 0.1-0.5 million | 681 | 16 | 114 |
| 0.5-1 million | 69 | 9 | 13 |
| 1-5 million | 41 | 13 | 10 |
| 5 million+ | 11 | 2 | 0 |

Larger cities are expected to be polycentric more often; a strong pattern the other way would be worth checking.

## GHSL vs OSM tests

Where both give a clear type (507 cities) they agree for 93%.

| ghsl_type | monocentric | polycentric |
|---|---|---|
| monocentric | 473 | 33 |
| polycentric | 0 | 1 |

## OpenStreetMap mapping quality

| osm_quality | cities |
|---|---|
| good | 711 |
| sparse | 200 |
| very sparse | 68 |

POIs per km2 of built-up area, against the median city. Sparse cities lean on GHSL alone.

## Network groups (tile mode) vs types

| network group | monocentric | polycentric | unclear |
|---|---|---|---|
| 0 | 100 | 13 | 58 |
| 1 | 119 | 2 | 12 |
| 2 | 134 | 0 | 5 |
| 3 | 65 | 3 | 5 |
| 4 | 340 | 22 | 55 |

- Adjusted Rand index with the types: -0.01 (0 = chance, 1 = identical)
- Rank correlation of the network's main score with the number of GHSL subcentres: 0.30 (p = 2.67e-20)

Mapping-quality check (network group vs OSM mapping quality):

| network group | good | sparse | very sparse |
|---|---|---|---|
| 0 | 65 | 67 | 39 |
| 1 | 95 | 31 | 7 |
| 2 | 118 | 19 | 2 |
| 3 | 54 | 14 | 5 |
| 4 | 347 | 59 | 11 |

- **Warning:** the network groups follow OSM mapping quality (group(s) 0 are mostly sparsely mapped cities); Kruskal-Wallis p = 6.67e-27. These groups partly sort cities by mapping effort, not by shape.

## Network groups (city mode) vs types

| network group | monocentric | polycentric | unclear |
|---|---|---|---|
| 0 | 637 | 14 | 37 |
| 1 | 165 | 26 | 100 |

- Adjusted Rand index with the types: 0.11 (0 = chance, 1 = identical)
- Rank correlation of the network's main score with the number of GHSL subcentres: 0.46 (p = 6.65e-53)

Mapping-quality check (network group vs OSM mapping quality):

| network group | good | sparse | very sparse |
|---|---|---|---|
| 0 | 524 | 142 | 22 |
| 1 | 187 | 58 | 46 |

- No sign that the groups follow OSM mapping quality (Kruskal-Wallis p = 0.0265).

## Largest polycentric cities

| label | pop | ghsl_gs_subcentres | ghsl_mcmillen_subcentres | confidence |
|---|---|---|---|---|
| Hajipur, India | 9755302.728 | 1 | 0 | low |
| Lucknow, India | 5214964.391 | 1 | 0 | low |
| Kanpur, India | 4413624.598 | 1 | 0 | low |
| Nagpur, India | 3399891.104 | 1 | 0 | low |
| Kalyan-Dombivli, India | 3214860.965 | 1 | 0 | low |
| Ponnani, India | 2751608.997 | 2 | 0 | low |
| Tamluk, India | 2451036.828 | 2 | 0 | low |
| Prayagraj, India | 2358994.904 | 1 | 0 | low |
| Visakhapatnam, India | 1634439.651 | 1 | 0 | low |
| Dhulian, India | 1566770.357 | 2 | 0 | low |
| Vasai-Virar, India | 1551310.96 | 3 | 0 | low |
| Gyanpur, India | 1324131.508 | 1 | 0 | low |
| Amritsar, India | 1303219.344 | 1 | 0 | low |
| Begusarai, India | 1193507.773 | 2 | 0 | low |
| Colachel, India | 1001954.461 | 3 | 0 | low |

## Very large urban centres (check by hand)

The GHSL database joins built-up areas that touch, so some of these are several towns counted as one city. Their type says more about the region than about one city.

| label | pop | area_km2 | final_type |
|---|---|---|---|
| Hajipur, India | 9755302.728 | 3152.8 | polycentric |

## Caveats

- The classical tests here count built-up area (GHSL) or POIs, not jobs, and use relative cut-offs (see analysis/classical.py).
- GHSL built-up is from satellite imagery for one year (the epoch downloaded); OSM is today's map.
- Network groups are a check: they are learned without labels and need not line up with the two classical types.
