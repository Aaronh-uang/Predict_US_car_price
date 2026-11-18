// In-browser evaluator for the exported LightGBM model.
// Mirrors LightGBM's own decision rules (tree.h) so predictions match Python
// exactly; tests/parity.mjs checks this against LightGBM's output.

const K_ZERO = 1e-35;

function compile(tree) {
  if (!tree.f) return { leafOnly: tree.v[0] };
  return {
    f: Int32Array.from(tree.f),
    t: Float64Array.from(tree.t),
    g: Uint8Array.from(tree.g),
    l: Int32Array.from(tree.l),
    r: Int32Array.from(tree.r),
    v: Float64Array.from(tree.v),
    c: (tree.c || []).map((d) => {
      // category sets are delta-encoded: [first, gap, gap, ...]
      let acc = 0;
      return new Set(d.map((g) => (acc += g)));
    }),
  };
}

function evalTree(tr, x) {
  if (tr.leafOnly !== undefined) return tr.leafOnly;
  let n = 0;
  while (n >= 0) {
    const flag = tr.g[n];
    let v = x[tr.f[n]];
    if (flag & 1) {
      // categorical: NaN or negative goes right; listed codes go left
      if (Number.isNaN(v) || v < 0) n = tr.r[n];
      else n = tr.c[tr.t[n]].has(Math.trunc(v)) ? tr.l[n] : tr.r[n];
    } else {
      const missing = (flag >> 2) & 3; // 0 none, 1 zero, 2 nan
      if (Number.isNaN(v) && missing !== 2) v = 0;
      if ((missing === 1 && v >= -K_ZERO && v <= K_ZERO) || (missing === 2 && Number.isNaN(v))) {
        n = flag & 2 ? tr.l[n] : tr.r[n];
      } else {
        n = v <= tr.t[n] ? tr.l[n] : tr.r[n];
      }
    }
  }
  return tr.v[~n];
}

export class Booster {
  constructor(json) {
    this.json = json;
    this.trees = json.trees.map(compile);
    this.lookup = {};
    for (const col of json.categorical) {
      this.lookup[col] = new Map(json.vocab[col].map((v, i) => [v, i]));
    }
  }

  raw(x) {
    let s = 0;
    for (const tr of this.trees) s += evalTree(tr, x);
    return s;
  }

  /** Feature object (strings for categories, numbers or null) -> Float64Array. */
  vector(input) {
    const { features, categorical } = this.json;
    const x = new Float64Array(features.length);
    features.forEach((name, j) => {
      const val = input[name];
      if (categorical.includes(name)) {
        const code = val == null ? undefined : this.lookup[name].get(String(val));
        x[j] = code === undefined ? NaN : code;
      } else {
        x[j] = val == null || val === "" || !Number.isFinite(Number(val)) ? NaN : Number(val);
      }
    });
    return x;
  }

  /** Price-range bucket from the calibration split (split-conformal by price level). */
  range(raw, unknown = false) {
    const { edges, lo, hi } = (unknown && this.json.interval_unknown) || this.json.interval;
    let b = 0;
    while (b < edges.length && edges[b] <= raw) b++;
    return [Math.exp(raw + lo[b]), Math.exp(raw + hi[b])];
  }
}

export class PriceModel {
  constructor(model, calibration, meta) {
    this.booster = new Booster(model);
    this.calibration = calibration;
    this.meta = meta;
  }

  static async load(base = "data/") {
    const get = (f) => fetch(base + f).then((r) => (r.ok ? r.json() : null));
    const [model, calibration, meta] = await Promise.all([
      get("model.json"), get("calibration.json"), get("meta.json"),
    ]);
    if (!model) throw new Error("model.json missing - run the training pipeline first");
    return new PriceModel(model, calibration, meta);
  }

  /** Multiplier from 2020-market to today's market for this car. */
  marketFactor(make, modelKey, age) {
    const c = this.calibration;
    if (!c) return 1;
    const a = Math.max(0, Math.min(c.age_factor.length - 1, Math.round(age)));
    // offsets are per age band ([under 8, 8-15, 16+] years); older files use one number
    const band = (c.age_bands || []).filter((b) => age >= b).length;
    const pick = (v) => (Array.isArray(v) ? v[band] ?? 0 : v || 0);
    return Math.exp(c.age_factor[a] + pick(c.make_offset[make]) + pick(c.model_offset[modelKey]));
  }

  /**
   * car: {make, model, trim, year, mileage, owner_count, has_accidents,
   *       frame_damaged, salvage, body_type, ...specs}
   * when: fractional year the price is for (defaults to now)
   * market: "today" (calibrated) or "2020" (raw model)
   */
  predict(car, { market = "today", when = currentYearFrac() } = {}) {
    const age = when - car.year;
    const modelKey = `${car.make}|${car.model}`;
    const x = this.booster.vector({
      ...car,
      age,
      make: car.make,
      model_key: modelKey,
      trim_key: car.trim ? `${modelKey}|${car.trim}` : null,
    });
    const raw = this.booster.raw(x);
    const factor = market === "today" ? this.marketFactor(car.make, modelKey, age) : 1;
    // Without a ZIP or options the estimate is less certain: use the wider range table.
    const opts = this.booster.json.options || [];
    const unknown = car.lat == null || opts.some((o) => car[o.flag] == null);
    const [low, high] = this.booster.range(raw, unknown);
    return { price: Math.exp(raw) * factor, low: low * factor, high: high * factor, factor, age };
  }
}

export function currentYearFrac(d = new Date()) {
  const start = Date.UTC(d.getUTCFullYear(), 0, 1);
  const day = Math.floor((d.getTime() - start) / 86400000);
  return d.getUTCFullYear() + day / 365.25;
}
