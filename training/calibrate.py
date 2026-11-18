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
import re
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
AGE_BANDS = [8, 16]     # offsets for cars under 8, 8-15 and 16+ years old


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


# ForCar and the 2020 data name a few models differently.
ALIASES = {
    ("mercedesbenz", "mlclass"): "M-Class",
    ("mercedesbenz", "glclass"): "GL-Class",
    ("mini", "hardtop"): "Cooper",
    ("mini", "clubman"): "Cooper Clubman",
    ("mini", "countryman"): "Countryman",
}


def nkey(s) -> str:
    return norm_key(str(s).replace("&", " and "))


def match_models(catalog, make_raw, model_raw, makes_by_norm):
    """ForCar (make, model) -> (catalog make, [catalog models]).

    ForCar often names a family ("RX", "Silverado", "F-450") where the 2020
    data has the versions ("RX 350", "Silverado 1500", "F-450 Super Duty").
    Then the ForCar price applies to the whole family, weighted by listings.
    """
    mk_n, mo = norm_key(make_raw), str(model_raw).strip()
    if mk_n == "toyota" and mo.lower().startswith("scion "):      # "Toyota Scion xB"
        mk_n, mo = "scion", mo[6:]
    make = makes_by_norm.get(mk_n)
    if make is None:
        return None, []
    models = catalog[make]
    mo = ALIASES.get((mk_n, norm_key(mo)), mo)
    if make == "BMW" and (m := re.fullmatch(r"M?(\d)\d\d[a-zA-Z]*", mo)):   # 328i -> 3 Series
        mo = f"{m[1]} Series"
    if make == "Ford" and re.fullmatch(r"E-?\d50", mo):                     # E-350 -> E-Series
        mo = "E-Series"

    by_norm = {nkey(m): m for m in models}
    exact = by_norm.get(nkey(mo))
    low = mo.lower()
    family = [m for m in models if m != exact and (m.lower().startswith(low + " ") or m.lower().startswith(low + "-"))]
    if exact:
        family_n = sum(models[m]["n"] for m in family)
        return make, [exact] if models[exact]["n"] >= family_n else [exact] + family
    if family:
        return make, family
    near = [m for k, m in by_norm.items() if k.startswith(nkey(mo)) or nkey(mo).startswith(k)]
    return make, near[:1] if len(near) == 1 else []


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
        make, models = match_models(catalog, r.make, r.model, makes_by_norm)
        if not models:
            continue
        age = max(asof_frac - float(r.model_year), -1.0)
        total_n = sum(catalog[make][m]["n"] for m in models)
        rows, w = [], []
        for m in models:
            rm, wm = rows_for(make, m, catalog[make][m], int(r.model_year), float(r.mileage), age)
            share = catalog[make][m]["n"] / total_n
            rows += rm
            w += [share * x / sum(wm) for x in wm]
        recs.append({"make": make, "model_keys": {f"{make}|{m}": catalog[make][m]["n"] / total_n for m in models},
                     "age": age, "listings": float(r.listings), "price": float(r.price), "w": np.array(w, float)})
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

    # 2) make, then model, offsets on what's left - shrunk toward zero, and
    #    separate for young / mid / old cars (a model's 2009s can move very
    #    differently from its 2019s)
    d["band"] = np.searchsorted(AGE_BANDS, d["age"].to_numpy(), side="right")
    g = d.groupby(["make", "band"])
    make_off = (g.apply(lambda x: np.sum(x["listings"] * x["res"]), include_groups=False)
                / (g["listings"].sum() + SHRINK_MAKE))
    d["res2"] = d["res"] - make_off.reindex(pd.MultiIndex.from_arrays([d["make"], d["band"]])).to_numpy()

    ex = d.assign(model_key=d["model_keys"].map(lambda m: list(m.items()))).explode("model_key")
    ex["share"] = ex["model_key"].str[1]
    ex["model_key"] = ex["model_key"].str[0]
    ex["wl"] = ex["listings"] * ex["share"]
    g = ex.groupby(["model_key", "band"])
    model_off = (g.apply(lambda x: np.sum(x["wl"] * x["res2"]), include_groups=False)
                 / (g["wl"].sum() + SHRINK_MODEL))

    d["family_offset"] = [sum(sh * model_off.get((k, b), 0.0) for k, sh in m.items())
                          for m, b in zip(d["model_keys"], d["band"])]
    d.drop(columns=["model_keys"]).assign(models=d["model_keys"].map(lambda m: ";".join(m))).round(4).to_csv(
        ARTIFACTS / "calibration_rows.csv", index=False)     # per-row working, for checking
    fitted = age_factor.to_numpy()[d["age_i"]] + (d["res"] - d["res2"]) + d["family_offset"]

    def by_band(series):
        out = {}
        for (key, band), v in series.items():
            out.setdefault(key, [0.0] * (len(AGE_BANDS) + 1))[int(band)] = round(float(v), 5)
        return out

    before = wmedian(np.abs(d["lr"]), d["listings"])
    after = wmedian(np.abs(d["lr"] - fitted), d["listings"])

    write_json(WEB_DATA / "calibration.json", {
        "asof": asof.isoformat(),
        "source": "ForCar open data - US used-car asking prices by model and model year",
        "url": "https://forcar.org/open-data/",
        "license": "CC BY 4.0",
        "age_factor": [round(float(v), 5) for v in age_factor],
        "age_bands": AGE_BANDS,
        "make_offset": by_band(make_off),
        "model_offset": by_band(model_off),
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
