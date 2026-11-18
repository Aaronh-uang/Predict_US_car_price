"""Step 4: calibrate 2020-market predictions to today's market.

The model learned how price depends on mileage, age, trim and history from
2020 listings. Prices have moved a lot since, and not evenly across models.
ForCar publishes today's average asking price per make / model / model year
(CC BY 4.0, rebuilt daily). For every ForCar row we ask the model what that
car would have listed for in 2020 at the same age and average mileage; the
ratio becomes a correction:

    today = 2020 prediction x exp(age_factor[age] + make_offset + model_offset)

Offsets are shrunk toward zero when a make/model has few listings, so thin
data can't swing a price.

    python training/calibrate.py                   # downloads today's ForCar file
    python training/calibrate.py --forcar data/forcar.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import urllib.error
import urllib.request

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import ARTIFACTS, DATA_DIR, FEATURES, WEB_DATA, encode, norm_key, write_json

FORCAR_URL = "https://forcar.org/datasets/used-car-prices-by-model-year.csv"
MAX_AGE = 30
SHRINK_MAKE = 400     # listings needed before a make's own offset gets half weight
SHRINK_MODEL = 200
CLIP = np.log([0.5, 2.5])
TOP_TRIMS = 5


def download(path) -> None:
    req = urllib.request.Request(FORCAR_URL, headers={"User-Agent": "car-price-predictor (research)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    except urllib.error.URLError as e:
        # python.org builds on macOS ship without CA certificates; curl uses the system's.
        print(f"  urllib failed ({e.reason}); retrying with curl")
        subprocess.run(["curl", "-fsSL", "--max-time", "120", "-o", str(path), FORCAR_URL], check=True)


def wmedian(x, w):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, c[-1] / 2)])


def load_catalog():
    idx = json.loads((WEB_DATA / "catalog" / "index.json").read_text())
    cat = {}
    for m in idx["makes"]:
        cat[m["make"]] = json.loads((WEB_DATA / "catalog" / m["file"]).read_text())["models"]
    return cat


def match_model(catalog, make_norm, model_norm, makes_by_norm):
    make = makes_by_norm.get(make_norm)
    if make is None:
        return None, None
    models = {norm_key(m): m for m in catalog[make]}
    if model_norm in models:
        return make, models[model_norm]
    near = [m for k, m in models.items() if k.startswith(model_norm) or model_norm.startswith(k)]
    return (make, near[0]) if len(near) == 1 else (make, None)


def rows_for(make, model, entry, model_year, mileage, age):
    """The same rows the website would build for this car, one per common trim."""
    trims = [t for t in entry["trims"] if t["years"][1] >= min(model_year, entry["years"][1]) - 2]
    trims = (trims or entry["trims"])[:TOP_TRIMS]
    choices = [(t["name"], t["specs"], t["n"]) for t in trims] or [(None, entry["specs"], 1)]
    rows, weights = [], []
    for trim, specs, n in choices:
        rows.append({
            "age": age, "mileage": mileage, "make": make,
            "model_key": f"{make}|{model}",
            "trim_key": f"{make}|{model}|{trim}" if trim else None,
            **{k: specs.get(k) for k in ["body_type", "wheel_system", "fuel_type", "transmission",
                                         "engine_type", "horsepower", "engine_displacement",
                                         "city_fuel_economy", "highway_fuel_economy"]},
            "owner_count": None, "has_accidents": None, "frame_damaged": None, "salvage": None,
        })
        weights.append(n)
    return rows, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", help="use a local ForCar CSV instead of downloading")
    ap.add_argument("--asof", help="date the ForCar file describes (YYYY-MM-DD); default today")
    args = ap.parse_args()

    if args.forcar:
        path = args.forcar
    else:
        path = DATA_DIR / f"forcar_{dt.date.today():%Y%m%d}.csv"
        if not path.exists():
            print(f"Downloading {FORCAR_URL}")
            download(path)
    asof = dt.date.fromisoformat(args.asof) if args.asof else dt.date.today()
    asof_frac = asof.year + (asof.timetuple().tm_yday - 1) / 365.25

    fc = pd.read_csv(path)
    fc = fc.rename(columns={"avg_price_usd": "price", "avg_mileage": "mileage"})
    fc = fc.dropna(subset=["make", "model", "model_year", "listings", "price", "mileage"])
    fc = fc[(fc["listings"] >= 5) & (fc["price"] > 500)]

    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    vocab = meta["vocab"]
    booster = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    catalog = load_catalog()
    makes_by_norm = {norm_key(m): m for m in catalog}

    all_rows, owner, recs = [], [], []
    for r in fc.itertuples(index=False):
        make, model = match_model(catalog, norm_key(r.make), norm_key(r.model), makes_by_norm)
        if model is None:
            continue
        age = max(asof_frac - float(r.model_year), -1.0)
        rows, w = rows_for(make, model, catalog[make][model], int(r.model_year), float(r.mileage), age)
        recs.append({"make": make, "model_key": f"{make}|{model}", "age": age,
                     "listings": float(r.listings), "price": float(r.price), "w": np.array(w, float)})
        owner += [len(recs) - 1] * len(rows)
        all_rows += rows

    if not recs:
        raise SystemExit("No ForCar rows matched the catalog - check make/model names.")

    X = encode(pd.DataFrame(all_rows, columns=FEATURES), vocab)
    pred = np.exp(booster.predict(X))
    owner = np.array(owner)
    for i, rec in enumerate(recs):
        p = pred[owner == i]
        rec["pred2020"] = float(np.sum(p * rec["w"]) / rec["w"].sum())
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "w"} for r in recs])
    d["lr"] = np.clip(np.log(d["price"] / d["pred2020"]), *CLIP)
    d["age_i"] = d["age"].round().clip(0, MAX_AGE).astype(int)

    # 1) market-wide change by vehicle age, lightly smoothed
    by_age = d.groupby("age_i").apply(lambda g: wmedian(g["lr"], g["listings"]), include_groups=False)
    by_age = by_age.reindex(range(MAX_AGE + 1)).interpolate(limit_direction="both")
    age_factor = by_age.rolling(3, center=True, min_periods=1).mean()
    d["res"] = d["lr"] - age_factor.to_numpy()[d["age_i"]]

    # 2) make, then model, offsets on what's left - shrunk toward zero
    g = d.groupby("make")
    make_off = (g.apply(lambda x: np.sum(x["listings"] * x["res"]), include_groups=False)
                / (g["listings"].sum() + SHRINK_MAKE))
    d["res2"] = d["res"] - d["make"].map(make_off)
    g = d.groupby("model_key")
    model_off = (g.apply(lambda x: np.sum(x["listings"] * x["res2"]), include_groups=False)
                 / (g["listings"].sum() + SHRINK_MODEL))

    fitted = age_factor.to_numpy()[d["age_i"]] + d["make"].map(make_off) + d["model_key"].map(model_off)
    before = wmedian(np.abs(d["lr"]), d["listings"])
    after = wmedian(np.abs(d["lr"] - fitted), d["listings"])

    write_json(WEB_DATA / "calibration.json", {
        "asof": asof.isoformat(),
        "source": "ForCar open data - US used-car asking prices by model and model year",
        "url": "https://forcar.org/open-data/",
        "license": "CC BY 4.0",
        "age_factor": [round(float(v), 5) for v in age_factor],
        "make_offset": {k: round(float(v), 5) for k, v in make_off.items()},
        "model_offset": {k: round(float(v), 5) for k, v in model_off.items()},
        "fit": {
            "rows_matched": int(len(d)), "rows_total": int(len(fc)),
            "listings_matched": int(d["listings"].sum()), "listings_total": int(fc["listings"].sum()),
            "median_abs_gap_before_pct": round(100 * (np.exp(before) - 1), 1),
            "median_abs_gap_after_pct": round(100 * (np.exp(after) - 1), 1),
        },
    }, compact=False)

    print(f"Matched {len(d):,} of {len(fc):,} ForCar rows "
          f"({d['listings'].sum() / fc['listings'].sum():.0%} of listings)")
    print(f"Market change vs 2020 by age: " + ", ".join(
        f"{a}y {np.exp(age_factor[a]) - 1:+.0%}" for a in [1, 3, 5, 8, 12] if a <= MAX_AGE))
    print(f"Median gap to ForCar averages: {np.exp(before) - 1:.1%} uncorrected -> {np.exp(after) - 1:.1%} calibrated")
    print(f"Wrote {WEB_DATA / 'calibration.json'}")


if __name__ == "__main__":
    main()
