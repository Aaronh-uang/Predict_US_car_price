// VIN decoding through NHTSA's free vPIC API (no key needed), then matching
// the result to the make / model / trim names the model was trained on.

const VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/";

export const norm = (s) => String(s ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");

export function isVin(s) {
  return /^[A-HJ-NPR-Z0-9]{17}$/i.test(String(s).trim());
}

export async function decodeVin(vin) {
  const res = await fetch(`${VPIC}${encodeURIComponent(vin.trim())}?format=json`);
  if (!res.ok) throw new Error(`NHTSA lookup failed (${res.status})`);
  const r = (await res.json()).Results?.[0];
  if (!r || !r.Make) throw new Error(r?.ErrorText || "NHTSA couldn't decode this VIN");
  const num = (v) => (v === "" || v == null || !Number.isFinite(Number(v)) ? null : Number(v));
  return {
    make: r.Make,
    model: r.Model,
    year: num(r.ModelYear),
    trimLabel: r.Trim || "",
    trimText: [r.Trim, r.Trim2, r.Series, r.Series2].filter(Boolean).join(" "),
    horsepower: num(r.EngineHP),
    engine_displacement: num(r.DisplacementCC) ?? (num(r.DisplacementL) ? num(r.DisplacementL) * 1000 : null),
    cylinders: num(r.EngineCylinders),
    config: r.EngineConfiguration || "",
    drive: r.DriveType || "",
    fuel: r.FuelTypePrimary || "",
    electrification: r.ElectrificationLevel || "",
    body: r.BodyClass || "",
    transmission: r.TransmissionStyle || "",
    warning: r.ErrorCode && r.ErrorCode !== "0" ? r.ErrorText : "",
  };
}

/** Pick the catalog make whose normalized name matches the decoded one. */
export function matchMake(makes, decoded) {
  const n = norm(decoded.make);
  return makes.find((m) => norm(m.make) === n) || makes.find((m) => n.startsWith(norm(m.make)) || norm(m.make).startsWith(n));
}

export function matchModel(models, decoded) {
  const n = norm(decoded.model);
  const names = Object.keys(models);
  return (
    names.find((m) => norm(m) === n) ||
    names.filter((m) => norm(m).startsWith(n) || n.startsWith(norm(m))).sort((a, b) => models[b].n - models[a].n)[0]
  );
}

/** Trim with the most words in common with NHTSA's trim/series text. */
export function matchTrim(trims, decoded) {
  const words = new Set(decoded.trimText.toLowerCase().split(/[^a-z0-9.]+/).filter(Boolean));
  if (!words.size) return null;
  let best = null, bestScore = 0;
  for (const t of trims) {
    const tw = t.name.toLowerCase().split(/[^a-z0-9.]+/).filter(Boolean);
    const hit = tw.filter((w) => words.has(w)).length;
    const score = hit / Math.max(tw.length, 1) + hit * 0.01 + Math.log10(t.n) * 0.001;
    if (hit && score > bestScore) { best = t; bestScore = score; }
  }
  return best;
}

/** NHTSA wording -> the categories used in training. */
export function mapSpecs(d) {
  const out = {};
  if (d.horsepower) out.horsepower = d.horsepower;
  if (d.engine_displacement) out.engine_displacement = Math.round(d.engine_displacement / 100) * 100;
  const drive = d.drive.toLowerCase();
  if (/awd|all[- ]wheel/.test(drive)) out.wheel_system = "AWD";
  else if (/4wd|4x4|four[- ]wheel|4-wheel/.test(drive)) out.wheel_system = "4WD";
  else if (/fwd|front[- ]wheel/.test(drive)) out.wheel_system = "FWD";
  else if (/rwd|rear[- ]wheel/.test(drive)) out.wheel_system = "RWD";
  const fuel = `${d.fuel} ${d.electrification}`.toLowerCase();
  if (/bev|battery electric/.test(fuel) || fuel.trim() === "electric") out.fuel_type = "Electric";
  else if (/hybrid|hev|phev/.test(fuel)) out.fuel_type = "Hybrid";
  else if (/diesel/.test(fuel)) out.fuel_type = "Diesel";
  else if (/flex/.test(fuel)) out.fuel_type = "Flex Fuel Vehicle";
  else if (/gasoline/.test(fuel)) out.fuel_type = "Gasoline";
  const tr = d.transmission.toLowerCase();
  if (/cvt|continuously/.test(tr)) out.transmission = "CVT";
  else if (/dual|dct/.test(tr)) out.transmission = "Dual Clutch";
  else if (/manual/.test(tr)) out.transmission = "Manual";
  else if (/automatic/.test(tr)) out.transmission = "Automatic";
  return out;
}
