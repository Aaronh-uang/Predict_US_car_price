// Checks that the browser evaluator reproduces LightGBM's predictions.
//   node tests/parity.mjs
import { readFileSync } from "node:fs";
import { Booster } from "../web/js/model.js";

const model = JSON.parse(readFileSync(new URL("../web/data/model.json", import.meta.url)));
const fx = JSON.parse(readFileSync(new URL("../artifacts/parity_fixture.json", import.meta.url)));
const b = new Booster(model);

let worst = 0;
fx.X.forEach((row, i) => {
  const x = Float64Array.from(row, (v) => (v === null ? NaN : v));
  worst = Math.max(worst, Math.abs(b.raw(x) - fx.raw[i]));
});
// 1e-6 in log-price is 0.0001% of the price.
const ok = worst < 1e-6;
console.log(`${fx.X.length} rows, worst |JS - LightGBM| = ${worst.toExponential(2)} (log price) -> ${ok ? "PASS" : "FAIL"}`);
process.exit(ok ? 0 : 1);
