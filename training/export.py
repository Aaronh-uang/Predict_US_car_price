"""Step 3: turn the trained model into files the static website loads.

    web/data/model.json            trees + category vocab + likely-range table
    web/data/meta.json             accuracy numbers, data coverage
    web/data/catalog/index.json    makes -> file names
    web/data/catalog/<make>.json   models -> trims -> typical specs (auto-fill)

    python training/export.py
"""
from __future__ import annotations

import json
import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (ARTIFACTS, LISTINGS, SPEC_CATEGORICAL, SPEC_NUMERIC, WEB_DATA,
                    slug, write_json)

MIN_TRIM_LISTINGS = 10


def sig(x: float, digits: int = 8) -> float:
    """Round leaf values; the change is far below a cent on the final price."""
    return float(f"{x:.{digits}g}")


def compact_tree(tree: dict) -> dict:
    """Flatten one LightGBM tree into parallel arrays.

    Child pointers: >= 0 is an internal node index, < 0 is leaf ~index.
    flags: bit0 categorical, bit1 default_left, bits2-3 missing type (0 None, 1 Zero, 2 NaN).
    For categorical nodes, t holds an index into `c` (the sets of codes that go left).
    """
    f, t, flags, left, right, values, cats = [], [], [], [], [], [], []
    missing = {"None": 0, "Zero": 1, "NaN": 2}

    def walk(node) -> int:
        if "leaf_index" in node or "split_index" not in node:
            values.append(sig(node.get("leaf_value", 0.0)))
            return ~(len(values) - 1)
        i = len(f)
        f.append(node["split_feature"])
        flag = (2 if node["default_left"] else 0) | (missing[node["missing_type"]] << 2)
        if node["decision_type"] == "==":
            flag |= 1
            codes = sorted(int(c) for c in str(node["threshold"]).split("||"))
            cats.append([codes[0]] + [b - a for a, b in zip(codes, codes[1:])])   # delta-encoded
            t.append(len(cats) - 1)
        else:
            t.append(sig(node["threshold"], 12))
        flags.append(flag)
        left.append(None)
        right.append(None)
        left[i] = walk(node["left_child"])
        right[i] = walk(node["right_child"])
        return i

    root = tree["tree_structure"]
    if "split_index" not in root:            # a single-leaf tree
        return {"v": [sig(root.get("leaf_value", 0.0))]}
    walk(root)
    out = {"f": f, "t": t, "g": flags, "l": left, "r": right, "v": values}
    if cats:
        out["c"] = cats
    return out


def mode(s: pd.Series):
    s = s.dropna()
    return None if s.empty else str(s.value_counts().index[0])


def med(s: pd.Series):
    s = s.dropna()
    return None if s.empty else round(float(s.median()), 1)


def specs_of(g: pd.DataFrame) -> dict:
    out = {c: mode(g[c]) for c in SPEC_CATEGORICAL}
    out.update({c: med(g[c]) for c in SPEC_NUMERIC})
    return {k: v for k, v in out.items() if v is not None}


def build_catalog(df: pd.DataFrame) -> dict[str, dict]:
    """make -> {models: {model: {n, years, specs, trims: [...]}}}.

    Specs come from each trim's most recent model years, so a 2019 Camry SE
    auto-fills with the current generation's engine and mpg.
    """
    df = df.copy()
    df["trim_name"] = df["trim_name"].fillna("")
    latest = df.groupby(["make_name", "model_name", "trim_name"])["year"].transform("max")
    recent = df[df["year"] >= latest - 1]

    recent_by_trim = dict(list(recent.groupby(["make_name", "model_name", "trim_name"], sort=False)))

    catalog: dict[str, dict] = {}
    for (make, model), g in df.groupby(["make_name", "model_name"], sort=False):
        trims = []
        for trim, tg in g[g["trim_name"] != ""].groupby("trim_name", sort=False):
            if len(tg) < MIN_TRIM_LISTINGS:
                continue
            trims.append({
                "name": trim, "n": int(len(tg)),
                "years": [int(tg["year"].min()), int(tg["year"].max())],
                "specs": specs_of(recent_by_trim[(make, model, trim)]),
            })
        trims.sort(key=lambda d: -d["n"])
        newest = g[g["year"] >= g["year"].max() - 2]
        catalog.setdefault(make, {})[model] = {
            "n": int(len(g)),
            "years": [int(g["year"].min()), int(g["year"].max())],
            "specs": specs_of(newest),
            "trims": trims,
        }
    return catalog


def main():
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    dump = booster.dump_model()
    assert dump["feature_names"] == meta["features"], "feature order changed since training"

    trees = [compact_tree(t) for t in dump["tree_info"]]
    model = {
        "version": 2,
        "target": "log_price",
        "features": meta["features"],
        "numeric": meta["numeric"],
        "categorical": meta["categorical"],
        "vocab": meta["vocab"],
        "options": meta.get("options", []),
        "interval": meta["interval"],
        "interval_unknown": meta.get("interval_unknown", meta["interval"]),
        "trees": trees,
    }
    write_json(WEB_DATA / "model.json", model)
    write_json(WEB_DATA / "zip3.json", meta.get("zip3", {}))

    m = meta["metrics"]
    write_json(WEB_DATA / "meta.json", {
        "test": m["test"], "unknown": m.get("unknown", {}), "by_segment": m["by_segment"],
        "importance": m["importance"][:10], "trees": m["trees"], "data": meta["data"],
    }, compact=False)

    df = pd.read_parquet(LISTINGS)
    catalog = build_catalog(df)
    index = []
    for make in sorted(catalog, key=str.lower):
        models = catalog[make]
        file = f"{slug(make)}.json"
        write_json(WEB_DATA / "catalog" / file, {"make": make, "models": models})
        index.append({"make": make, "file": file, "n": sum(v["n"] for v in models.values())})
    write_json(WEB_DATA / "catalog" / "index.json", {"makes": index})

    size = (WEB_DATA / "model.json").stat().st_size / 1e6
    n_nodes = sum(len(t.get("f", [])) for t in trees)
    n_trims = sum(len(m["trims"]) for mk in catalog.values() for m in mk.values())
    import gzip
    gz = len(gzip.compress((WEB_DATA / "model.json").read_bytes())) / 1e6
    print(f"model.json: {len(trees)} trees, {n_nodes:,} splits, {size:.1f} MB ({gz:.1f} MB gzipped, what visitors download)")
    if gz > 4:
        print("  warning: that's heavy for phones - consider fewer trees (MAX_ROUNDS in train.py)")
    print(f"catalog: {len(index)} makes, {sum(len(v) for v in catalog.values()):,} models, {n_trims:,} trims")


if __name__ == "__main__":
    main()
