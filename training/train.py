"""Step 2: train the price model on data/listings.parquet.

LightGBM on log(price), with make/model/trim as real categorical features and
monotone constraints on mileage, age and history. A held-out calibration split
sets the "likely range" (split-conformal, by price level), and a separate test
split measures accuracy.

    python training/train.py
    python training/train.py --sample 300000     # quick run while iterating
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (ARTIFACTS, CATEGORICAL, FEATURES, LISTINGS, MONOTONE, NUMERIC,
                    add_keys, build_vocab, encode, split_of, write_json)

PARAMS = dict(
    objective="regression",
    learning_rate=0.1,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.85,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    cat_smooth=20,
    cat_l2=10,
    max_cat_threshold=24,
    min_data_per_group=50,
    monotone_constraints_method="intermediate",
    verbose=-1,
    seed=42,
    num_threads=0,
)
MAX_ROUNDS = 1200
INTERVAL = (0.10, 0.90)   # 80% "likely range"
N_BUCKETS = 12


def metrics(y_true, y_pred, lo=None, hi=None) -> dict:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    out = {
        "n": int(len(y_true)),
        "mae_usd": float(np.mean(np.abs(err))),
        "median_abs_error_usd": float(np.median(np.abs(err))),
        "rmse_usd": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(100 * np.mean(ape)),
        "median_ape_pct": float(100 * np.median(ape)),
        "within_10pct": float(100 * np.mean(ape <= 0.10)),
    }
    if lo is not None:
        out["range_coverage_pct"] = float(100 * np.mean((y_true >= lo) & (y_true <= hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="train on a random subset")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(LISTINGS)
    # Keep one market era: a handful of listings date back to the 2010s,
    # when prices were at a different level.
    old = df["listed_year"] < df["listed_year"].max() - 1
    if old.any():
        print(f"Dropping {int(old.sum()):,} listings from before {df['listed_year'].max() - 1}")
        df = df[~old]
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=0)
    df = add_keys(df)
    df["split"] = split_of(df["vin"])
    print(f"{len(df):,} listings: " + ", ".join(f"{k} {v:,}" for k, v in df["split"].value_counts().items()))

    vocab = build_vocab(df[df["split"] == "train"])
    for col in CATEGORICAL:
        print(f"  {col:<14} {len(vocab[col]):>6,} categories")

    X = encode(df, vocab)
    y = np.log(df["price"].to_numpy())
    part = {s: (df["split"] == s).to_numpy() for s in ["train", "valid", "calib", "test"]}

    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    params = dict(PARAMS, monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES])
    dtrain = lgb.Dataset(X[part["train"]], y[part["train"]], feature_name=FEATURES,
                         categorical_feature=cat_idx, free_raw_data=False)
    dvalid = lgb.Dataset(X[part["valid"]], y[part["valid"]], reference=dtrain)
    booster = lgb.train(
        params, dtrain, MAX_ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)],
    )
    print(f"Trained {booster.best_iteration} trees in {time.time() - t0:,.0f}s")

    # Likely range: residual quantiles on the calibration split, per price level.
    pred_cal = booster.predict(X[part["calib"]], num_iteration=booster.best_iteration)
    resid = y[part["calib"]] - pred_cal
    edges = np.quantile(pred_cal, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
    bucket = np.searchsorted(edges, pred_cal, side="right")
    q_lo = [float(np.quantile(resid[bucket == b], INTERVAL[0])) for b in range(N_BUCKETS)]
    q_hi = [float(np.quantile(resid[bucket == b], INTERVAL[1])) for b in range(N_BUCKETS)]

    # Accuracy on the untouched test split.
    pred_test = booster.predict(X[part["test"]], num_iteration=booster.best_iteration)
    b_test = np.searchsorted(edges, pred_test, side="right")
    y_test = df.loc[part["test"], "price"].to_numpy()
    p_test = np.exp(pred_test)
    lo = p_test * np.exp(np.take(q_lo, b_test))
    hi = p_test * np.exp(np.take(q_hi, b_test))
    m = metrics(y_test, p_test, lo, hi)

    seg = pd.cut(y_test, [0, 10_000, 20_000, 35_000, 60_000, np.inf],
                 labels=["<$10k", "$10–20k", "$20–35k", "$35–60k", "$60k+"])
    by_segment = {str(s): metrics(y_test[seg == s], p_test[seg == s]) for s in seg.categories if (seg == s).sum() > 0}

    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    importance = sorted(
        ({"feature": f, "share_pct": float(100 * g / gain.sum())} for f, g in zip(FEATURES, gain)),
        key=lambda d: -d["share_pct"])

    print("\nTest set:")
    for k, v in m.items():
        print(f"  {k:<22} {v:,.2f}")
    print("Top features:", ", ".join(f"{d['feature']} {d['share_pct']:.0f}%" for d in importance[:6]))

    ARTIFACTS.mkdir(exist_ok=True)
    booster.save_model(str(ARTIFACTS / "model.txt"), num_iteration=booster.best_iteration)
    write_json(ARTIFACTS / "meta.json", {
        "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
        "vocab": vocab,
        "interval": {"level": INTERVAL, "edges": edges.tolist(), "lo": q_lo, "hi": q_hi},
        "metrics": {"test": m, "by_segment": by_segment, "importance": importance,
                    "trees": booster.best_iteration},
        "data": {
            "listings": int(len(df)),
            "listed_years": [int(df["listed_year"].min()), int(df["listed_year"].max())],
            "model_years": [int(df["year"].min()), int(df["year"].max())],
        },
    }, compact=False)

    # A few hundred test rows + LightGBM's own predictions, so the website's
    # JavaScript evaluator can be checked against it (tests/parity.mjs).
    n = min(500, int(part["test"].sum()))
    idx = np.flatnonzero(part["test"])[:n]
    Xs = X[idx].copy()
    rng = np.random.default_rng(0)
    Xs[rng.random(Xs.shape) < 0.08] = np.nan        # exercise missing-value paths
    write_json(ARTIFACTS / "parity_fixture.json", {
        "X": [[None if np.isnan(v) else float(v) for v in row] for row in Xs],
        "raw": booster.predict(Xs, num_iteration=booster.best_iteration).tolist(),
    })
    print(f"\nSaved to {ARTIFACTS}/  ({time.time() - t0:,.0f}s total)")


if __name__ == "__main__":
    main()
