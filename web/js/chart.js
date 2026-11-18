// Small dependency-free SVG line chart: one series, an 80% range band,
// a marker for "your car", and a crosshair + tooltip on hover.

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(v);
  return ticks;
}

/**
 * points: [{x, y, lo, hi}] sorted by x
 * opts: {marker: {x, y}, xFormat, yFormat, xTitle, tooltip(point) -> [title, lines...]}
 */
export function renderChart(container, points, opts) {
  container.innerHTML = "";
  const W = Math.max(container.clientWidth, 280);
  const H = Math.round(Math.min(300, Math.max(220, W * 0.5)));
  const m = { t: 16, r: 16, b: 40, l: 64 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": opts.label || "Price chart" }, container);

  const xs = points.map((p) => p.x);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yMax = Math.max(...points.map((p) => p.hi ?? p.y)) * 1.05;
  const yTicks = niceTicks(0, yMax);
  if (yTicks[yTicks.length - 1] < yMax) yTicks.push(yTicks[yTicks.length - 1] + (yTicks[1] - yTicks[0]));
  const yTop = yTicks[yTicks.length - 1];
  const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - (v / yTop) * (H - m.t - m.b);

  // grid + axes (recessive)
  const grid = el("g", { class: "grid" }, svg);
  for (const t of yTicks) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }, grid);
    const lab = el("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end", class: "tick" }, grid);
    lab.textContent = opts.yFormat(t);
  }
  const xTicks = opts.xTicks || niceTicks(x0, x1, Math.max(3, Math.floor(W / 110)));
  for (const t of xTicks) {
    const lab = el("text", { x: sx(t), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, grid);
    lab.textContent = opts.xFormat(t);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), class: "baseline" }, svg);
  if (opts.xTitle) {
    const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, svg);
    t.textContent = opts.xTitle;
  }

  // range band + line
  const band = points.map((p) => `${sx(p.x)},${sy(p.hi)}`).join(" ") + " " +
    points.slice().reverse().map((p) => `${sx(p.x)},${sy(p.lo)}`).join(" ");
  el("polygon", { points: band, class: "band" }, svg);
  el("polyline", { points: points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" "), class: "line" }, svg);

  if (opts.marker) {
    const mk = opts.marker;
    el("circle", { cx: sx(mk.x), cy: sy(mk.y), r: 6, class: "marker" }, svg);
    const t = el("text", { x: sx(mk.x), y: sy(mk.y) - 14, "text-anchor": sx(mk.x) > W - 80 ? "end" : sx(mk.x) < m.l + 40 ? "start" : "middle", class: "marker-label" }, svg);
    t.textContent = opts.marker.label || "your car";
  }

  // hover layer: crosshair snaps to the nearest x
  const hair = el("line", { y1: m.t, y2: H - m.b, class: "hair", visibility: "hidden" }, svg);
  const dot = el("circle", { r: 4.5, class: "hover-dot", visibility: "hidden" }, svg);
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  container.appendChild(tip);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);

  const show = (evt) => {
    const box = svg.getBoundingClientRect();
    const px = ((evt.clientX - box.left) / box.width) * W;
    let best = points[0];
    for (const p of points) if (Math.abs(sx(p.x) - px) < Math.abs(sx(best.x) - px)) best = p;
    const X = sx(best.x), Y = sy(best.y);
    hair.setAttribute("x1", X); hair.setAttribute("x2", X); hair.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X); dot.setAttribute("cy", Y); dot.setAttribute("visibility", "visible");
    const [title, ...lines] = opts.tooltip(best);
    tip.innerHTML = `<strong>${title}</strong>` + lines.map((l) => `<span>${l}</span>`).join("");
    tip.hidden = false;
    const left = (X / W) * box.width;
    tip.style.left = `${Math.min(Math.max(left, 70), box.width - 70)}px`;
    tip.style.top = `${(Y / H) * box.height}px`;
  };
  const hide = () => { hair.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerdown", show);
  hit.addEventListener("pointerleave", hide);
}
