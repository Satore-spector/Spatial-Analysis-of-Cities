"""Shared helpers for the Python half of the pipeline (steps 2-5).

Everything here is plain Python + numpy/pandas so the other modules stay short.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Paths ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def raster_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "rasters"


def analysis_dir(data_dir: Path, mode: str) -> Path:
    """Outputs of steps 3-5 live in data/analysis/<mode>/ (mode = city or tile)."""
    return Path(data_dir) / "analysis" / mode


# ---- City names -------------------------------------------------------------


def city_slug(city: str) -> str:
    """Same rule as city_slug() in the R scripts: 'Bengaluru, India' -> 'bengaluru'."""
    return re.sub(r"[^a-z]", "", city.split(",")[0].lower())


def find_cities(data_dir: Path, cities: list[str] | None = None) -> list[str]:
    """City slugs to process.

    With no list, every city that has all four OSM files in data/ is used
    (the R smoke test's 'smoketest' boundary has no POIs, so it is skipped).
    """
    data_dir = Path(data_dir)
    if cities:
        return [city_slug(c) for c in cities]
    slugs = []
    for f in sorted(data_dir.glob("*_boundary.gpkg")):
        slug = f.name[: -len("_boundary.gpkg")]
        needed = [data_dir / f"{slug}_{p}.gpkg" for p in ("pois", "buildings", "roads")]
        if all(p.exists() for p in needed):
            slugs.append(slug)
    return slugs


# ---- Channels (layers) of each city grid -----------------------------------

# POI categories. The rules below copy data_extraction/amenity_wrap.R exactly,
# including the order (the first rule that matches wins). If you change one,
# change the other.
POI_CATEGORIES = [
    "Food & drink", "Retail", "Finance", "Offices", "Health", "Education",
    "Worship", "Transport", "Leisure & culture", "Other",
]

_FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "biergarten"}
_FINANCE = {"bank", "atm", "bureau_de_change", "money_transfer"}
_HEALTH = {"hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary"}
_EDUCATION = {"school", "college", "university", "kindergarten", "library", "coaching", "training"}
_TRANSPORT = {"bus_station", "taxi", "fuel", "parking", "bicycle_rental", "charging_station"}
_LEISURE = {"cinema", "theatre", "community_centre", "arts_centre", "marketplace"}


def poi_category(pois: pd.DataFrame) -> pd.Series:
    """Vectorised version of the case_when() in amenity_wrap.R."""
    def has(col):
        return pois[col].notna() if col in pois else pd.Series(False, index=pois.index)

    amen = pois["amenity"] if "amenity" in pois else pd.Series(None, index=pois.index)
    rules = [
        ("Food & drink", amen.isin(_FOOD)),
        ("Retail", has("shop")),
        ("Finance", amen.isin(_FINANCE)),
        ("Offices", has("office")),
        ("Health", amen.isin(_HEALTH) | has("healthcare")),
        ("Education", amen.isin(_EDUCATION)),
        ("Worship", amen == "place_of_worship"),
        ("Transport", has("public_transport") | has("railway") | amen.isin(_TRANSPORT)),
        ("Leisure & culture", has("leisure") | has("tourism") | amen.isin(_LEISURE)),
    ]
    out = pd.Series("Other", index=pois.index, dtype=object)
    done = pd.Series(False, index=pois.index)
    for name, mask in rules:
        take = mask.fillna(False) & ~done
        out[take] = name
        done |= take
    return out


def channel_name(category: str) -> str:
    """'Food & drink' -> 'poi_food_drink' (safe for file and column names)."""
    return "poi_" + re.sub(r"[^a-z]+", "_", category.lower()).strip("_")


POI_CHANNELS = [channel_name(c) for c in POI_CATEGORIES]
# How each channel is combined when cells are merged into bigger cells:
# counts and lengths add up, fractions are averaged.
CHANNELS = POI_CHANNELS + ["buildings", "roads_major", "roads_minor", "built_up"]
CHANNEL_AGG = {**{c: "sum" for c in POI_CHANNELS},
               "buildings": "mean", "roads_major": "sum", "roads_minor": "sum",
               "built_up": "mean"}
CHANNEL_UNITS = {**{c: "POIs per cell" for c in POI_CHANNELS},
                 "buildings": "share of cell covered by building footprints",
                 "roads_major": "metres of major road per cell",
                 "roads_minor": "metres of minor road per cell",
                 "built_up": "share of cell built up (GHSL, satellite)"}

MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary",
               "motorway_link", "trunk_link", "primary_link", "secondary_link"}


# ---- Saved city grids -------------------------------------------------------


def save_city_grid(path: Path, grid: np.ndarray, mask: np.ndarray, meta: dict) -> None:
    np.savez_compressed(path, grid=grid.astype(np.float32), mask=mask.astype(np.uint8),
                        meta=json.dumps(meta))


def load_city_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        return z["grid"], z["mask"].astype(bool), json.loads(str(z["meta"]))


def block_reduce(a: np.ndarray, factor: int, how: str) -> np.ndarray:
    """Merge factor x factor blocks of a 2-D array (edges are trimmed)."""
    if factor == 1:
        return a.copy()
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    b = a[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return b.sum(axis=(1, 3)) if how == "sum" else b.mean(axis=(1, 3))


def md_table(df: pd.DataFrame) -> str:
    """A pandas table as a Markdown table (index included), without extra packages."""
    d = df.reset_index()
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def say(*parts) -> None:
    print(*parts, flush=True)
