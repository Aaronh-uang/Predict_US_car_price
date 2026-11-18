"""Step 5: how good are today's-market estimates? Score them against KBB.

Rebrowser publishes a free daily sample of AutoTrader listings
(github.com/rebrowser/autotrader-dataset, research and non-commercial use
with attribution). Listing prices are hidden in the sample, but each row
carries Kelley Blue Book's fair-price range, an independent estimate of what
the car is worth today. This script:

1. downloads the latest sample (last ~30 days) and adds its used listings to
   benchmark/autotrader_kbb.csv.gz, a slim file that grows week by week;
2. prices every listing seen in the last 30 days the way the website does
   (nationwide, options "Not sure", today's market) and compares with the
   middle of KBB's range;
3. writes web/data/benchmark.json (shown on the site) with a history of
   past runs.

    python training/benchmark.py
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import calibrate as C
from common import ARTIFACTS, ROOT, WEB_DATA, encode, market_factor, norm_key, write_json

SAMPLE_REPO = "https://github.com/rebrowser/autotrader-dataset"
STORE = ROOT / "benchmark" / "autotrader_kbb.csv.gz"
WINDOW_DAYS = 60     # score listings seen in the last 60 days of the store
KEEP = ["listingId", "seen", "year", "make", "model", "trim", "mileage", "listingType",
        "zip", "kbb_low", "kbb_high"]


def fetch_sample() -> pd.DataFrame:
    """Shallow, sparse clone of the sample repo: only the daily parquet files."""
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", SAMPLE_REPO, str(tmp / "s")],
                       check=True)
        subprocess.run(["git", "-C", str(tmp / "s"), "sparse-checkout", "set", "car-listings/data"], check=True)
        files = sorted((tmp / "s" / "car-listings" / "data").glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    df = df[df["listingType"].isin(["Used", "Certified", "Third-Party Certified"])
            & (pd.to_numeric(df["kbbFairPriceLow"], errors="coerce") > 500)
            & (pd.to_numeric(df["mileage"], errors="coerce") > 100)]
    return pd.DataFrame({
        "listingId": df["listingId"].astype(str),
        "seen": pd.to_datetime(df["_lastSeenAt"], utc=True).dt.strftime("%Y-%m-%d"),
        "year": df["year"].astype(int), "make": df["makeName"], "model": df["modelName"], "trim": df["trim"],
        "mileage": pd.to_numeric(df["mileage"]), "listingType": df["listingType"],
        "zip": df["sellerZip"].astype(str).str[:5],
        "kbb_low": df["kbbFairPriceLow"], "kbb_high": df["kbbFairPriceHigh"],
    })


def update_store(new: pd.DataFrame) -> pd.DataFrame:
    STORE.parent.mkdir(exist_ok=True)
    old = pd.read_csv(STORE, dtype={"listingId": str, "zip": str}) if STORE.exists() else pd.DataFrame(columns=KEEP)
    allr = pd.concat([old, new[KEEP]], ignore_index=True)
    allr = allr.sort_values("seen").drop_duplicates("listingId", keep="last")
    allr.to_csv(STORE, index=False, compression="gzip")
    print(f"benchmark store: {len(allr):,} listings ({len(allr) - len(old):+,} new), "
          f"{allr['seen'].min()} to {allr['seen'].max()}")
    return allr


def score(df: pd.DataFrame) -> dict:
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    cal = json.loads((WEB_DATA / "calibration.json").read_text())
    catalog = C.load_catalog()
    makes_by_norm = {norm_key(m): m for m in catalog}

    rows, info = [], []
    for r in df.itertuples(index=False):
        make, models = C.match_models(catalog, r.make, r.model, makes_by_norm)
        if not models:
            continue
        model = max(models, key=lambda m: catalog[make][m]["n"])
        entry = catalog[make][model]
        trim = None
        if isinstance(r.trim, str) and r.trim:
            k = norm_key(r.trim)
            trim = next((t for t in entry["trims"] if norm_key(t["name"]) == k), None) or \
                next((t for t in entry["trims"] if norm_key(t["name"]).startswith(k)), None)
        seen = pd.Timestamp(r.seen)
        age = seen.year + (seen.day_of_year - 1) / 365.25 - r.year
        specs = (trim or entry)["specs"]
        rows.append({
            "age": age, "mileage": r.mileage, "make": make, "model_key": f"{make}|{model}",
            "trim_key": f"{make}|{model}|{trim['name']}" if trim else None,
            "is_cpo": float(r.listingType != "Used"),
            **{k: specs.get(k) for k in ["body_type", "wheel_system", "fuel_type", "transmission", "engine_type",
                                         "horsepower", "engine_displacement", "city_fuel_economy", "highway_fuel_economy"]},
        })
        info.append((r.year, (r.kbb_low + r.kbb_high) / 2, market_factor(cal, make, f"{make}|{model}", age)))

    X = encode(pd.DataFrame(rows), meta["vocab"], meta["features"])
    raw = booster.predict(X)
    year, kbb, fac = (np.array(v, float) for v in zip(*info))
    pred = np.exp(raw) * fac
    err = pred / kbb - 1

    # Is the "likely range" (no ZIP / options) wide enough for today's market?
    iv = meta.get("interval_unknown", meta["interval"])
    b = np.searchsorted(iv["edges"], raw, side="right")
    lo, hi = pred * np.exp(np.take(iv["lo"], b)), pred * np.exp(np.take(iv["hi"], b))
    inside = (kbb >= lo) & (kbb <= hi)

    # How much wider must the range be (in log terms) to hold KBB 80% of the
    # time? The site applies this to today's-market ranges only.
    d = np.log(kbb / pred)
    half = np.where(d < 0, -np.take(iv["lo"], b), np.take(iv["hi"], b))
    scale = float(max(1.0, np.quantile(np.abs(d) / half, 0.80)))

    def summary(mask):
        e = err[mask]
        return {"n": int(mask.sum()), "median_gap_pct": round(float(100 * np.median(np.abs(e))), 1),
                "median_signed_pct": round(float(100 * np.median(e)), 1),
                "within_10pct": round(float(100 * np.mean(np.abs(e) <= 0.10)), 1),
                "kbb_inside_range_pct": round(float(100 * inside[mask].mean()), 1)}

    everything = np.ones(len(err), bool)
    return {"all": summary(everything), "matched_of": int(len(df)), "today_range_scale": round(scale, 3),
            "model_years_2015_2021": summary((year >= 2015) & (year <= 2021)),
            "model_years_2022_plus": summary(year >= 2022)}


def main():
    store = update_store(fetch_sample())
    cutoff = (pd.Timestamp(store["seen"].max()) - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    recent = store[store["seen"] >= cutoff]
    result = score(recent)
    out = WEB_DATA / "benchmark.json"
    history = json.loads(out.read_text()).get("history", []) if out.exists() else []
    today = dt.date.today().isoformat()
    history = [h for h in history if h["date"] != today] + [{"date": today, **{k: result["all"][k] for k in ("n", "median_gap_pct")}}]
    write_json(out, {
        "asof": today,
        "window": [cutoff, store["seen"].max()],
        "source": "Rebrowser AutoTrader sample (KBB fair-price range), github.com/rebrowser/autotrader-dataset",
        "how": "Nationwide price, options not set, today's market, compared with the middle of KBB's fair-price range.",
        **result,
        "history": history[-104:],
    }, compact=False)
    a = result["all"]
    print(f"Today vs KBB ({a['n']:,} listings, {cutoff} to {store['seen'].max()}): median gap {a['median_gap_pct']}% "
          f"(we're {a['median_signed_pct']:+}%), within 10%: {a['within_10pct']}%, "
          f"KBB inside our likely range: {a['kbb_inside_range_pct']}% -> today's ranges widened x{result['today_range_scale']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
