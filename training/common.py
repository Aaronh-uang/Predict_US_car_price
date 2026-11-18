"""Shared settings and helpers for the training pipeline.

Every script imports from here so the feature list, the raw-column mapping and
the key format can only be defined once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # raw + intermediate files (git-ignored)
ARTIFACTS = ROOT / "artifacts"      # trained model + metadata (git-ignored)
WEB_DATA = ROOT / "web" / "data"    # what the website loads (committed)

RAW_CSV = DATA_DIR / "used_cars_data.csv"
LISTINGS = DATA_DIR / "listings.parquet"

# Raw columns read from the 9 GB Kaggle CSV. Everything else is skipped while
# parsing, which keeps memory well under 16 GB.
RAW_COLUMNS = [
    "vin", "price", "year", "mileage", "listed_date", "is_new",
    "make_name", "model_name", "trim_name",
    "body_type", "wheel_system", "fuel_type", "transmission", "engine_type",
    "engine_displacement", "horsepower", "city_fuel_economy", "highway_fuel_economy",
    "has_accidents", "frame_damaged", "salvage", "owner_count",
    "major_options", "is_cpo", "is_oemcpo", "dealer_zip", "latitude", "longitude", "city",
]

# Model inputs, in the exact order the website builds its feature vector:
# NUMERIC, then one 0/1 flag per common option (chosen from the data at
# training time, saved in meta.json), then CATEGORICAL.
NUMERIC = [
    "age",                   # years between model year and listing date (fractional)
    "mileage",
    "horsepower",
    "engine_displacement",   # cc
    "city_fuel_economy",
    "highway_fuel_economy",
    "owner_count",
    "has_accidents",         # 1 / 0 / missing
    "frame_damaged",
    "salvage",
    "is_cpo",                # certified pre-owned 1 / 0
    "lat",                   # centre of the buyer's 3-digit ZIP area; missing = anywhere
    "lon",
]
CATEGORICAL = [
    "make",
    "model_key",     # "Make|Model"
    "trim_key",      # "Make|Model|Trim"
    "body_type",
    "wheel_system",
    "fuel_type",
    "transmission",
    "engine_type",
]
FEATURES = NUMERIC + CATEGORICAL     # without option flags; see feature_list()

# Options: keep the common ones (seen on 4-96% of listings), at most this many.
MAX_OPTIONS = 30
# Share of training rows whose location / options are hidden so the model
# learns a sensible answer for "not sure" and "anywhere".
MASK_UNKNOWN = 0.15
ZIP3_MIN_LISTINGS = 20

# Price can only go down as these go up (all else equal). This keeps the
# depreciation curves on the website smooth and sensible.
MONOTONE = {
    "age": -1, "mileage": -1, "owner_count": -1,
    "has_accidents": -1, "frame_damaged": -1, "salvage": -1,
}

# A category needs this many listings to get its own value; rarer ones fall
# back to "unknown" so the model leans on the broader make/model.
MIN_COUNT = {"trim_key": 30, "model_key": 20}
MIN_COUNT_DEFAULT = 20

TRANSMISSION_MAP = {
    "a": "Automatic", "automatic": "Automatic",
    "m": "Manual", "manual": "Manual",
    "cvt": "CVT",
    "dual clutch": "Dual Clutch", "dct": "Dual Clutch",
}

# Specs the website auto-fills from the make/model/trim catalog.
SPEC_CATEGORICAL = ["body_type", "wheel_system", "fuel_type", "transmission", "engine_type"]
SPEC_NUMERIC = ["horsepower", "engine_displacement", "city_fuel_economy", "highway_fuel_economy"]


def norm_key(s: str) -> str:
    """Loose matching key: 'F-150' == 'f150' == 'F 150'."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["make"] = df["make_name"].astype("string")
    df["model_key"] = df["make"] + "|" + df["model_name"].astype("string")
    trim = df["trim_name"].astype("string").fillna("")
    df["trim_key"] = (df["model_key"] + "|" + trim).where(trim != "")
    return df


def build_vocab(df: pd.DataFrame) -> dict[str, list[str]]:
    """Category -> integer code, most frequent first. Rare values are dropped."""
    vocab = {}
    for col in CATEGORICAL:
        counts = df[col].value_counts(dropna=True)
        counts = counts[counts >= MIN_COUNT.get(col, MIN_COUNT_DEFAULT)]
        vocab[col] = [str(v) for v in counts.index]
    return vocab


def feature_list(option_flags: list[str]) -> list[str]:
    return NUMERIC + option_flags + CATEGORICAL


def option_flag(label: str) -> str:
    return "opt_" + slug(label).replace("-", "_")


def choose_options(options: pd.Series) -> list[str]:
    """Most common options, as labels. `options` holds 'a|b|c' strings or missing."""
    have = options.dropna()
    counts = have.str.split("|").explode().value_counts()
    share = counts / max(len(have), 1)
    keep = share[(share >= 0.04) & (share <= 0.96)]
    return [str(v) for v in keep.index[:MAX_OPTIONS]]


def add_option_flags(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """One 0/1 column per option; missing when the listing has no options data."""
    df = df.copy()
    wrapped = "|" + df["options"].astype("string") + "|"
    known = df["options"].notna().to_numpy()
    for label in labels:
        hit = wrapped.str.contains("|" + label + "|", regex=False).fillna(False).to_numpy(dtype=float)
        df[option_flag(label)] = np.where(known, hit, np.nan)
    return df


def zip3_table(df: pd.DataFrame) -> dict[str, list]:
    """3-digit ZIP -> [lat, lon, most common city]. Only areas with enough listings."""
    d = df.dropna(subset=["zip3", "lat", "lon"])
    g = d.groupby("zip3")
    n = g.size()
    out = {}
    for z, grp in g:
        if n[z] < ZIP3_MIN_LISTINGS:
            continue
        city = grp["city"].mode()
        out[str(z)] = [round(float(grp["lat"].median()), 3), round(float(grp["lon"].median()), 3),
                       str(city.iloc[0]) if len(city) else ""]
    return out


def add_location(df: pd.DataFrame, table: dict[str, list]) -> pd.DataFrame:
    """Replace each listing's exact position with its ZIP3 area centre, as the site does."""
    df = df.copy()
    z = df["zip3"].astype("object")
    df["lat"] = z.map(lambda k: table[k][0] if k in table else np.nan).astype(float)
    df["lon"] = z.map(lambda k: table[k][1] if k in table else np.nan).astype(float)
    return df


def encode(df: pd.DataFrame, vocab: dict[str, list[str]], features: list[str] = FEATURES) -> np.ndarray:
    """DataFrame -> float matrix in `features` order. Unknown categories become NaN."""
    X = np.empty((len(df), len(features)), dtype=np.float64)
    for j, col in enumerate(features):
        if col in CATEGORICAL:
            lookup = {v: i for i, v in enumerate(vocab[col])}
            vals = df[col].astype("object").map(lookup) if col in df else pd.Series(np.nan, index=df.index)
        else:
            vals = df[col] if col in df else pd.Series(np.nan, index=df.index)
        X[:, j] = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return X


def market_factor(cal: dict, make: str, model_key: str, age: float) -> float:
    """2020 -> today multiplier from calibration.json (mirrors web/js/model.js)."""
    import math
    a = max(0, min(len(cal["age_factor"]) - 1, round(age)))
    band = sum(age >= b for b in cal.get("age_bands", []))
    pick = lambda v: v[band] if isinstance(v, list) else (v or 0.0)
    return math.exp(cal["age_factor"][a] + pick(cal["make_offset"].get(make, 0)) + pick(cal["model_offset"].get(model_key, 0)))


def split_of(vin: pd.Series) -> pd.Series:
    """Deterministic split from the VIN hash: 80 train / 5 valid / 5 calib / 10 test."""
    h = pd.util.hash_pandas_object(vin.astype("string"), index=False).to_numpy() % 100
    return pd.Series(
        np.select([h < 80, h < 85, h < 90], ["train", "valid", "calib"], "test"),
        index=vin.index,
    )


def write_json(path: Path, obj, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if compact:
            json.dump(obj, f, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(obj, f, indent=2, allow_nan=False)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
