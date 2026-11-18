import { PriceModel, currentYearFrac } from "./model.js";
import { renderChart } from "./chart.js";
import { isVin, decodeVin, matchMake, matchModel, matchTrim, mapSpecs } from "./vin.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const nice = (v) => usd.format(v >= 10000 ? Math.round(v / 100) * 100 : Math.round(v / 50) * 50);
const compactUsd = (v) => (v >= 1000 ? `$${int.format(v / 1000)}k` : usd.format(v));
const SPEC_CAT = ["body_type", "wheel_system", "fuel_type", "transmission", "engine_type"];
const SPEC_NUM = ["horsepower", "engine_displacement", "city_fuel_economy", "highway_fuel_economy"];
const NOW = currentYearFrac();
const THIS_YEAR = new Date().getFullYear();

const state = { market: "today", chart: "mileage", makes: [], makeData: {}, pm: null };

// ---------- catalog ----------
async function loadMake(make) {
  if (!state.makeData[make]) {
    const entry = state.makes.find((m) => m.make === make);
    state.makeData[make] = (await fetch(`data/catalog/${entry.file}`).then((r) => r.json())).models;
  }
  return state.makeData[make];
}
const models = () => state.makeData[$("#make").value] || {};
const modelEntry = () => models()[$("#model").value];
const trimEntry = () => modelEntry()?.trims.find((t) => t.name === $("#trim").value);

function fillSelect(sel, options, value) {
  sel.innerHTML = "";
  for (const o of options) {
    if (o.group) {
      const g = document.createElement("optgroup");
      g.label = o.group;
      o.items.forEach((it) => g.appendChild(new Option(it.label, it.value)));
      sel.appendChild(g);
    } else sel.appendChild(new Option(o.label, o.value));
  }
  if (value !== undefined && [...sel.options].some((o) => o.value === value)) sel.value = value;
}

function fillModels(value) {
  const ms = models();
  const names = Object.keys(ms).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const top = value ?? names.slice().sort((a, b) => ms[b].n - ms[a].n)[0];
  fillSelect($("#model"), names.map((n) => ({ label: n, value: n })), top);
}

function fillTrims(value) {
  const m = modelEntry();
  const year = Number($("#year").value);
  const opts = [{ label: "Not sure / not listed", value: "" }];
  if (m) {
    const target = Math.min(year, m.years[1]);
    const fits = (t) => t.years[0] - 1 <= target && target <= t.years[1] + 1;
    const near = m.trims.filter(fits), other = m.trims.filter((t) => !fits(t));
    const item = (t) => ({ label: t.name, value: t.name });
    const nearLabel = year > m.years[1] ? `Latest trims in data (${m.years[1]})` : `Sold around ${year}`;
    if (near.length) opts.push({ group: nearLabel, items: near.map(item) });
    if (other.length) opts.push({ group: "Other years", items: other.map(item) });
    if (value === undefined) value = near[0]?.name ?? "";
  }
  fillSelect($("#trim"), opts, value ?? "");
}

function applySpecs(specs = {}) {
  for (const k of SPEC_CAT) {
    const sel = document.querySelector(`[data-spec="${k}"]`);
    const v = specs[k] ?? "";
    if (v && ![...sel.options].some((o) => o.value === v)) sel.appendChild(new Option(v, v));
    sel.value = v;
  }
  for (const k of SPEC_NUM) {
    const v = specs[k];
    document.querySelector(`[data-spec="${k}"]`).value = v == null ? "" : int.format(v).replace(/,/g, "");
  }
}
const autofill = () => applySpecs((trimEntry() || modelEntry())?.specs);

// ---------- read the form ----------
function readCar() {
  const num = (v) => (v === "" || v == null ? null : Number(String(v).replace(/[^0-9.]/g, "")));
  const title = $("#title").value;
  const car = {
    year: Number($("#year").value),
    make: $("#make").value,
    model: $("#model").value,
    trim: $("#trim").value || null,
    mileage: num($("#mileage").value) ?? 0,
    owner_count: num($("#owners").value),
    has_accidents: num($("#accidents").value),
    salvage: title === "salvage" ? 1 : title ? 0 : null,
    frame_damaged: title === "frame" ? 1 : title === "clean" ? 0 : null,
  };
  for (const k of SPEC_CAT) car[k] = document.querySelector(`[data-spec="${k}"]`).value || null;
  for (const k of SPEC_NUM) car[k] = num(document.querySelector(`[data-spec="${k}"]`).value);
  return car;
}

// ---------- render ----------
function update() {
  const pm = state.pm;
  const car = readCar();
  if (!car.make || !car.model) return;
  const p = pm.predict(car, { market: state.market });

  $("#car-name").textContent = `${car.year} ${car.make} ${car.model}${car.trim ? " " + car.trim : ""} · ${int.format(car.mileage)} mi`;
  $("#price").textContent = nice(p.price);
  $("#range").innerHTML = `Likely <strong>${nice(p.low)} – ${nice(p.high)}</strong> <span class="muted">· 8 in 10 similar listings fall in this range</span>`;

  const cal = pm.calibration;
  $("#market-note").textContent = state.market === "today" && cal
    ? `2020 listing prices × ${p.factor.toFixed(2)}, matched to current asking prices for this model and age (ForCar, ${fmtDate(cal.asof)}).`
    : `What a car this age with these miles listed for in ${pm.meta?.data?.listed_years?.[1] ?? 2020}, before the 2021–22 price spike.`;

  const notes = [];
  const maxYear = pm.meta?.data?.model_years?.[1];
  if (maxYear && car.year > maxYear) notes.push(`Model year ${car.year} is newer than the training data (up to ${maxYear}). The estimate leans on age, mileage and the current-market adjustment.`);
  const key = `${car.make}|${car.model}`;
  if (!car.trim) notes.push("No trim selected, so this is priced as a typical " + car.model + ".");
  else if (!pm.booster.lookup.trim_key.has(`${key}|${car.trim}`)) notes.push(`Too few ${car.trim} listings to price that trim on its own; it's priced from the ${car.model} and its specs.`);
  if (state.market === "today" && cal && !(key in cal.model_offset)) notes.push(`No current asking prices for the ${car.model} in the ForCar data; using the ${car.make} and market-wide adjustment.`);
  if (car.salvage === 1) notes.push("Salvage and rebuilt titles are rare in the data, so treat this range as rough.");
  $("#notes").innerHTML = notes.map((n) => `<li>${n}</li>`).join("");

  drawChart(car, p);
  writeHash(car);
}

function drawChart(car, p) {
  const pm = state.pm;
  let points, marker, opts;
  if (state.chart === "mileage") {
    const maxMi = Math.max(150000, Math.ceil((car.mileage * 1.6) / 10000) * 10000);
    const step = maxMi / 40;
    points = Array.from({ length: 41 }, (_, i) => {
      const mi = i * step;
      const r = pm.predict({ ...car, mileage: mi }, { market: state.market });
      return { x: mi, y: r.price, lo: r.low, hi: r.high };
    });
    marker = { x: car.mileage, y: p.price, label: "your car" };
    opts = {
      xFormat: (v) => (v === 0 ? "0" : `${int.format(v / 1000)}k`),
      xTitle: "Mileage",
      tooltip: (pt) => [`${int.format(pt.x)} mi`, nice(pt.y), `range ${nice(pt.lo)} – ${nice(pt.hi)}`],
      label: `Estimated price of this ${car.model} from 0 to ${int.format(maxMi)} miles`,
    };
    $("#chart-title").textContent = "Value by mileage";
    $("#chart-note").textContent = "Same car at different mileages, everything else unchanged.";
  } else {
    const age = Math.max(NOW - car.year, 0.5);
    const perYear = Math.min(30000, Math.max(4000, car.mileage / age || 12000));
    points = Array.from({ length: 9 }, (_, k) => {
      const r = pm.predict({ ...car, mileage: car.mileage + k * perYear }, { market: state.market, when: NOW + k });
      return { x: THIS_YEAR + k, y: r.price, lo: r.low, hi: r.high, mi: car.mileage + k * perYear };
    });
    marker = { x: THIS_YEAR, y: p.price, label: "today" };
    opts = {
      xFormat: (v) => String(Math.round(v)),
      xTicks: points.filter((_, i) => i % 2 === 0).map((pt) => pt.x),
      xTitle: `Year · driving ${int.format(Math.round(perYear / 500) * 500)} mi/yr`,
      tooltip: (pt) => [`${pt.x}`, nice(pt.y), `${int.format(pt.mi)} mi`],
      label: `Projected value of this ${car.model} over the next 8 years`,
    };
    $("#chart-title").textContent = "Projected value";
    $("#chart-note").textContent = `Keeps your current pace of ${int.format(Math.round(perYear / 500) * 500)} miles a year and ${state.market === "today" ? "today's" : "2020"} market prices.`;
  }
  renderChart($("#chart"), points, { ...opts, marker, yFormat: compactUsd });

  const rows = state.chart === "mileage" ? points.filter((_, i) => i % 4 === 0) : points;
  $("#chart-table").innerHTML =
    `<tr><th>${state.chart === "mileage" ? "Mileage" : "Year"}</th><th>Estimate</th><th>Low</th><th>High</th></tr>` +
    rows.map((r) => `<tr><td>${state.chart === "mileage" ? int.format(r.x) : r.x}</td><td>${nice(r.y)}</td><td>${nice(r.lo)}</td><td>${nice(r.hi)}</td></tr>`).join("");
}

function renderAbout() {
  const { meta, calibration: cal } = state.pm;
  if (!meta) return;
  const t = meta.test;
  const d = meta.data;
  const top = meta.importance.slice(0, 4).map((f) => LABELS[f.feature] || f.feature).join(", ");
  $("#about").innerHTML = `
    <div class="stats">
      <div class="stat"><b>${t.median_ape_pct.toFixed(1)}%</b><span>median error on held-out cars</span></div>
      <div class="stat"><b>${t.within_10pct.toFixed(0)}%</b><span>of estimates within 10%</span></div>
      <div class="stat"><b>${usd.format(t.mae_usd)}</b><span>average error (MAE)</span></div>
      <div class="stat"><b>${t.range_coverage_pct.toFixed(0)}%</b><span>of cars inside the likely range</span></div>
    </div>
    <p>A gradient-boosted tree model (LightGBM, ${int.format(meta.trees)} trees) trained on ${int.format(d.listings)} used-car dealer listings from ${d.listed_years[0] === d.listed_years[1] ? d.listed_years[0] : d.listed_years.join("–")}, covering model years ${d.model_years[0]}–${d.model_years[1]}. Accuracy is measured on ${int.format(t.n)} cars the model never saw. The biggest drivers are ${top}. Price can only fall as mileage, age, owner count or damage go up, which keeps the curves above smooth.</p>
    <p>${cal ? `<strong>Today's market</strong> rescales the 2020 estimate for each model and age so it lines up with current average asking prices from ForCar (${int.format(cal.fit.listings_matched)} listings, ${fmtDate(cal.asof)}). On those averages, the typical gap drops from ${cal.fit.median_abs_gap_before_pct}% to ${cal.fit.median_abs_gap_after_pct}%. This is an adjustment from model-level averages, so it's less precise than the 2020 estimate.` : "Current-market calibration hasn't been run, so prices are in 2020 terms."}</p>
    <p>Everything runs in your browser: the model is a ${(JSON.stringify(state.pm.booster.json).length / 1e6).toFixed(1)} MB file, and nothing you enter is sent anywhere except a VIN, which goes to NHTSA's public decoder.</p>`;
}
const LABELS = {
  age: "age", mileage: "mileage", trim_key: "trim", model_key: "model", make: "make",
  horsepower: "horsepower", engine_displacement: "engine size", body_type: "body style",
  wheel_system: "drivetrain", fuel_type: "fuel type", engine_type: "engine", transmission: "transmission",
  city_fuel_economy: "city mpg", highway_fuel_economy: "highway mpg", owner_count: "owners",
  has_accidents: "accident history", frame_damaged: "frame damage", salvage: "salvage title",
};
const fmtDate = (iso) => new Date(iso + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

// ---------- VIN ----------
async function onVin() {
  const vin = $("#vin").value.trim().toUpperCase();
  const status = $("#vin-status");
  status.className = "status";
  if (!isVin(vin)) { status.className = "status error"; status.textContent = "A VIN is 17 letters and numbers (no I, O or Q)."; return; }
  $("#vin-go").disabled = true;
  status.textContent = "Looking up with NHTSA…";
  try {
    const d = await decodeVin(vin);
    const mk = matchMake(state.makes, d);
    if (!mk) throw new Error(`Decoded a ${d.year ?? ""} ${d.make} ${d.model}, but that make isn't in the training data.`);
    $("#make").value = mk.make;
    await loadMake(mk.make);
    const model = matchModel(models(), d);
    fillModels(model);
    if (d.year) setYear(d.year);
    fillTrims();
    const trim = model ? matchTrim(modelEntry().trims, d) : null;
    if (trim) $("#trim").value = trim.name;
    autofill();
    applySpecsPartial(mapSpecs(d));
    status.textContent = model
      ? `Decoded ${d.year} ${mk.make} ${model}${d.trimLabel ? " " + d.trimLabel : ""}. ${trim ? `Matched trim "${trim.name}".` : "Pick the closest trim."} Now enter the mileage.`
      : `Decoded ${d.year} ${mk.make} ${d.model}, but there's no "${d.model}" in the data. Pick the closest model.`;
    update();
    $("#mileage").focus();
  } catch (e) {
    status.className = "status error";
    status.textContent = e.message.includes("fetch") ? "Couldn't reach NHTSA. Check your connection and try again." : e.message;
  } finally {
    $("#vin-go").disabled = false;
  }
}
function applySpecsPartial(s) {
  for (const [k, v] of Object.entries(s)) {
    const elx = document.querySelector(`[data-spec="${k}"]`);
    if (!elx) continue;
    if (elx.tagName === "SELECT" && ![...elx.options].some((o) => o.value === v)) continue;
    elx.value = v;
  }
}
function setYear(y) {
  const sel = $("#year");
  if (![...sel.options].some((o) => Number(o.value) === y)) sel.appendChild(new Option(y, y));
  sel.value = String(y);
}

// ---------- share link ----------
function writeHash(car) {
  const q = new URLSearchParams();
  for (const k of ["year", "make", "model", "trim", "mileage", "owner_count", "has_accidents"]) if (car[k] != null) q.set(k, car[k]);
  if ($("#title").value) q.set("title", $("#title").value);
  history.replaceState(null, "", "#" + q.toString());
}
function readHash() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}

// ---------- setup ----------
async function init() {
  try {
    const [pm, idx] = await Promise.all([PriceModel.load(), fetch("data/catalog/index.json").then((r) => r.json())]);
    state.pm = pm;
    state.makes = idx.makes;
  } catch (e) {
    $("#price").textContent = "—";
    $("#range").textContent = "Couldn't load the model: " + e.message;
    return;
  }
  const pm = state.pm;
  if (!pm.calibration) {
    state.market = "2020";
    document.querySelector('[data-market="today"]').hidden = true;
  }

  fillSelect($("#year"), Array.from({ length: THIS_YEAR + 2 - 1995 }, (_, i) => {
    const y = THIS_YEAR + 1 - i; return { label: String(y), value: String(y) };
  }), String(THIS_YEAR - 5));
  fillSelect($("#make"), state.makes.map((m) => ({ label: m.make, value: m.make })));
  for (const k of SPEC_CAT) {
    fillSelect(document.querySelector(`[data-spec="${k}"]`),
      [{ label: "Not sure", value: "" }, ...pm.booster.json.vocab[k].slice().sort().map((v) => ({ label: v, value: v }))]);
  }

  const h = readHash();
  const startMake = state.makes.some((m) => m.make === h.make) ? h.make
    : state.makes.slice().sort((a, b) => b.n - a.n)[0].make;
  $("#make").value = startMake;
  await loadMake(startMake);
  if (h.year) setYear(Number(h.year));
  fillModels(h.model && models()[h.model] ? h.model : undefined);
  fillTrims(h.trim);
  autofill();
  $("#mileage").value = int.format(h.mileage ? Number(h.mileage) : 60000);
  if (h.owner_count) $("#owners").value = h.owner_count;
  if (h.has_accidents) $("#accidents").value = h.has_accidents;
  if (h.title) $("#title").value = h.title;

  $("#make").addEventListener("change", async () => { await loadMake($("#make").value); fillModels(); fillTrims(); autofill(); update(); });
  $("#model").addEventListener("change", () => { fillTrims(); autofill(); update(); });
  $("#year").addEventListener("change", () => { fillTrims($("#trim").value); update(); });
  $("#trim").addEventListener("change", () => { autofill(); update(); });
  $("#mileage").addEventListener("input", update);
  $("#mileage").addEventListener("blur", () => {
    const v = Number($("#mileage").value.replace(/[^0-9]/g, ""));
    $("#mileage").value = int.format(v || 0);
  });
  for (const id of ["#owners", "#accidents", "#title"]) $(id).addEventListener("change", update);
  $$("[data-spec]").forEach((e) => e.addEventListener(e.tagName === "SELECT" ? "change" : "input", () => {
    $("#specs-note").textContent = "edited";
    update();
  }));
  $("#vin-go").addEventListener("click", onVin);
  $("#vin").addEventListener("keydown", (e) => { if (e.key === "Enter") onVin(); });
  $$("[data-market]").forEach((b) => b.addEventListener("click", () => {
    state.market = b.dataset.market;
    $$("[data-market]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    update();
  }));
  $$("[data-chart]").forEach((b) => b.addEventListener("click", () => {
    state.chart = b.dataset.chart;
    $$("[data-chart]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    update();
  }));
  let raf;
  window.addEventListener("resize", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(update); });

  $("#sources").innerHTML = `Training data: <a href="https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset">US Used Cars dataset</a> (Kaggle). ` +
    (pm.calibration ? `Current-market prices: <a href="https://forcar.org/open-data/">ForCar open data</a>, CC BY 4.0. ` : "") +
    `VIN decoding: <a href="https://vpic.nhtsa.dot.gov/api/">NHTSA vPIC</a>. ` +
    `<a href="https://github.com/Aaronh-uang/Predict_US_car_price">Source code</a>.`;

  renderAbout();
  update();
}

init();
