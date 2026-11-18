"""Step 1: raw Kaggle CSV (9 GB) -> clean used-car listings Parquet (~100 MB).

Reads only the ~20 columns the model needs, in chunks, so a 16 GB laptop
never holds the whole file in memory.

    python training/prep.py                      # expects data/used_cars_data.csv
    python training/prep.py --csv ~/Downloads/used_cars_data.csv
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import LISTINGS, RAW_COLUMNS, RAW_CSV, TRANSMISSION_MAP

CHUNK = 250_000

SCHEMA = pa.schema([
    ("vin", pa.string()), ("price", pa.float64()), ("year", pa.int32()),
    ("mileage", pa.float64()), ("age", pa.float64()), ("listed_year", pa.int32()),
    ("make_name", pa.string()), ("model_name", pa.string()), ("trim_name", pa.string()),
    ("body_type", pa.string()), ("wheel_system", pa.string()), ("fuel_type", pa.string()),
    ("transmission", pa.string()), ("engine_type", pa.string()),
    ("engine_displacement", pa.float64()), ("horsepower", pa.float64()),
    ("city_fuel_economy", pa.float64()), ("highway_fuel_economy", pa.float64()),
    ("owner_count", pa.float64()), ("has_accidents", pa.float64()),
    ("frame_damaged", pa.float64()), ("salvage", pa.float64()),
])

BOOL = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}


def clean_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    raw.columns = [c.strip().lower() for c in raw.columns]
    df = pd.DataFrame(index=raw.index)

    def num(col):
        return pd.to_numeric(raw[col], errors="coerce")

    def text(col):
        s = raw[col].astype("string").str.strip()
        return s.mask(s.isin(["", "--", "nan", "None"]))

    def flag(col):
        return raw[col].astype("string").str.strip().str.lower().map(BOOL).astype("float64")

    df["vin"] = text("vin")
    df["price"] = num("price")
    df["year"] = num("year")
    df["mileage"] = num("mileage")

    listed = pd.to_datetime(raw["listed_date"], errors="coerce")
    listed_frac = listed.dt.year + (listed.dt.day_of_year - 1) / 365.25
    df["age"] = (listed_frac - df["year"]).round(3)
    df["listed_year"] = listed.dt.year

    for col in ["make_name", "model_name", "trim_name", "body_type",
                "wheel_system", "fuel_type", "engine_type"]:
        df[col] = text(col)
    t = text("transmission").str.lower()
    df["transmission"] = t.map(TRANSMISSION_MAP).astype("string").fillna(t.str.title())

    for col in ["engine_displacement", "horsepower", "city_fuel_economy",
                "highway_fuel_economy", "owner_count"]:
        df[col] = num(col)
    for col in ["has_accidents", "frame_damaged", "salvage"]:
        df[col] = flag(col)

    is_new = flag("is_new")
    keep = (
        (is_new != 1)                                  # used cars only
        & df["price"].between(1_000, 300_000)
        & df["mileage"].between(100, 400_000)          # <100 mi is effectively new
        & (df["year"] >= 1995)
        & df["age"].between(-1.5, 40)
        & df["make_name"].notna() & df["model_name"].notna() & df["vin"].notna()
    )
    df = df[keep].copy()
    df["year"] = df["year"].astype("int32")
    df["listed_year"] = df["listed_year"].astype("int32")
    # physically impossible specs -> missing rather than dropping the row
    df.loc[~df["horsepower"].between(40, 1600), "horsepower"] = np.nan
    df.loc[~df["engine_displacement"].between(500, 9000), "engine_displacement"] = np.nan
    for col in ["city_fuel_economy", "highway_fuel_economy"]:
        df.loc[~df[col].between(5, 160), col] = np.nan
    df.loc[~df["owner_count"].between(1, 15), "owner_count"] = np.nan
    return df[[f.name for f in SCHEMA]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RAW_CSV))
    ap.add_argument("--out", default=str(LISTINGS))
    args = ap.parse_args()

    wanted = set(RAW_COLUMNS)
    tmp = args.out + ".tmp"
    writer = pq.ParquetWriter(tmp, SCHEMA)
    t0, seen, kept = time.time(), 0, 0
    reader = pd.read_csv(
        args.csv, usecols=lambda c: c.strip().lower() in wanted,
        dtype=str, chunksize=CHUNK, on_bad_lines="skip",
    )
    for raw in reader:
        seen += len(raw)
        df = clean_chunk(raw)
        kept += len(df)
        writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
        print(f"  {seen:>10,} rows read, {kept:>10,} kept  ({time.time() - t0:,.0f}s)", flush=True)
    writer.close()

    # The same car is often listed more than once: keep its latest listing.
    df = pd.read_parquet(tmp)
    before = len(df)
    df = df.sort_values("age").drop_duplicates("vin", keep="last")
    df.to_parquet(args.out, index=False)
    import os
    os.remove(tmp)
    print(f"\nDone: {len(df):,} used-car listings ({before - len(df):,} duplicate VINs removed) -> {args.out}")
    print(f"Listing dates: {df['listed_year'].min()}-{df['listed_year'].max()}, "
          f"model years {df['year'].min()}-{df['year'].max()}")


if __name__ == "__main__":
    main()
