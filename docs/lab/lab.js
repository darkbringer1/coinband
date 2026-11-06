// coinband lab: interactive replays of the risk-model results (data from experiments/build_lab.py).
"use strict";

const $ = (id) => document.getElementById(id);
const C = {
  bg: "#151b23", grid: "rgba(139,152,168,0.13)", axis: "#8b98a8", text: "#e6edf3", muted: "#8b98a8",
  price: "#e6edf3", timestral: "#3987e5", garch: "#d95926", blend: "#199e70", realized: "rgba(230,237,243,0.35)",
};
const NAMES = { timestral: "Timestral", garch: "GARCH", blend: "Blend" };
const cache = {};
const load = async (f) => (cache[f] ??= await (await fetch(`data/${f}`)).json());

const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;
const signed = (v, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}%`;
function fmtPrice(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (a >= 1) return v.toLocaleString("en-US", { maximumFractionDigits: 4 });
  return v.toPrecision(4);
}
const pad = (n) => String(n).padStart(2, "0");
function fmtDate(ms, withHour = false) {
  const d = new Date(ms);
  const day = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
  return withHour ? `${day} ${pad(d.getUTCHours())}:00 UTC` : day;
}
function niceStep(range, target) {
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

// Size a canvas to its CSS box and return a 2D context in CSS pixels.
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, r.width, r.height);
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  return { ctx, w: r.width, h: r.height };
}

function yAxis(ctx, P, Y, lo, hi, fmt) {
  const step = niceStep(hi - lo, 5);
  ctx.strokeStyle = C.grid;
  ctx.fillStyle = C.axis;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    ctx.beginPath(); ctx.moveTo(P.l, Y(v)); ctx.lineTo(P.r, Y(v)); ctx.stroke();
    ctx.fillText(fmt(v), P.l - 8, Y(v));
  }
}

function xDates(ctx, P, X, t0, t1, toMs, withHour = false) {
  const span = t1 - t0;
  const n = Math.max(2, Math.floor((P.r - P.l) / 110));
  ctx.fillStyle = C.axis;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let k = 0; k <= n; k++) {
    const t = t0 + (span * k) / n;
    const ms = toMs(t);
    ctx.fillText(withHour ? fmtDate(ms, true).slice(5, 16) : fmtDate(ms).slice(5), X(t), P.b + 6);
  }
}

function tooltip(canvas, html, x, y) {
  const tip = canvas.parentElement.querySelector(".tip");
  if (html == null) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.innerHTML = html;
  const w = canvas.getBoundingClientRect().width;
  tip.style.left = `${x + 14 + tip.offsetWidth > w ? x - 14 - tip.offsetWidth : x + 14}px`;
  tip.style.top = `${Math.max(4, y - 10)}px`;
}
const sw = (c) => `<span class="sw" style="background:${c}"></span>`;

// ---------------------------------------------------------------- 1. risk lines

// playback: 1x advances 4 hours per second; 2x, 4x and 8x multiply that
const BASE_HOURS_PER_SECOND = 4;
const risk = { pair: null, level: "01", win: 168, end: 1, playing: false, data: null, hover: null, speed: 1, pos: 1, lastTs: 0 };

function riskStats(d, a, b, level) {
  const p = level === "01" ? 0.01 : 0.05;
  const out = {};
  for (const m of ["t", "g"]) {
    const line = d[m + level];
    let hits = 0, loss = 0;
    for (let i = a; i <= b; i++) {
      const diff = Math.log(d.close[i]) - Math.log(line[i]);
      if (diff < 0) hits++;
      loss += Math.max(p * diff, (p - 1) * diff);
    }
    const n = b - a + 1;
    out[m] = { hits, n, rate: hits / n, loss: (loss / n) * 1e4 };
  }
  return out;
}

function drawRisk() {
  const d = risk.data;
  if (!d) return;
  const canvas = $("risk-chart");
  const { ctx, w, h } = setup(canvas);
  const P = { l: 70, r: w - 86, t: 16, b: h - 26 };
  const n = d.close.length;
  const end = Math.max(1, Math.min(n - 1, risk.end));
  const a = Math.max(0, end - risk.win + 1);
  const t = (k) => d["t" + risk.level][k], g = (k) => d["g" + risk.level][k];
  let lo = Infinity, hi = -Infinity;
  for (let i = a; i <= end; i++) {
    lo = Math.min(lo, d.close[i], t(i), g(i));
    hi = Math.max(hi, d.close[i], d.last[i]);
  }
  const padY = (hi - lo) * 0.08 || hi * 0.01;
  lo -= padY; hi += padY;
  const X = (i) => P.l + ((i - a + 0.5) / (end - a + 1)) * (P.r - P.l);
  const Y = (v) => P.b - ((v - lo) / (hi - lo)) * (P.b - P.t);
  yAxis(ctx, P, Y, lo, hi, fmtPrice);
  xDates(ctx, P, X, a, end, (i) => d.t0 + Math.round(i) * d.step, end - a < 200);
  const bw = (P.r - P.l) / (end - a + 1);
  // the two risk lines, as steps (each hour has its own level)
  const stepLine = (get, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = a; i <= end; i++) {
      const x0 = X(i) - bw / 2, x1 = X(i) + bw / 2, y = Y(get(i));
      if (i === a) ctx.moveTo(x0, y); else ctx.lineTo(x0, y);
      ctx.lineTo(x1, y);
    }
    ctx.stroke();
  };
  stepLine(g, C.garch);
  stepLine(t, C.timestral);
  // price
  ctx.strokeStyle = C.price;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = a; i <= end; i++) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), Y(d.close[i]));
  ctx.stroke();
  // breaches: a dashed drop from the crossed line to the price, and a dot at the price in that model's color
  // (when both lines were crossed, the two dots sit side by side)
  const r = Math.max(3, Math.min(5.5, bw * 0.5));
  for (const [get, color, dx] of [[t, C.timestral, -1], [g, C.garch, 1]]) {
    for (let i = a; i <= end; i++) {
      if (d.close[i] >= get(i)) continue;
      const both = d.close[i] < t(i) && d.close[i] < g(i);
      const x = X(i) + (both ? dx * r : 0);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(x, Y(get(i))); ctx.lineTo(x, Y(d.close[i])); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, Y(d.close[i]), r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  // direct labels at the right end
  const labels = [[Y(t(end)), "Timestral", C.timestral], [Y(g(end)), "GARCH", C.garch], [Y(d.close[end]), "price", C.text]].sort((p, q) => p[0] - q[0]);
  let last = -Infinity;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  for (const [y0, text, color] of labels) {
    const y = Math.max(y0, last + 14);
    last = y;
    ctx.fillStyle = color;
    ctx.fillText(text, P.r + 6, Math.min(y, P.b - 4) + 0.5);
  }
  if (risk.hover != null && risk.hover >= a && risk.hover <= end) {
    const i = risk.hover;
    ctx.strokeStyle = "rgba(230,237,243,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(i), P.t); ctx.lineTo(X(i), P.b); ctx.stroke();
  }
  // stats: the window shown and everything up to its end
  const lvlTxt = risk.level === "01" ? "1%" : "5%";
  const win = riskStats(d, a, end, risk.level);
  const all = riskStats(d, 0, end, risk.level);
  const stat = (m, name, color) => `
    <div class="stat"><div class="name">${sw(color)}${name}</div>
      <div class="row"><span>Breaches in this window</span><b>${win[m].hits} of ${win[m].n} hours (${pct(win[m].rate)})</b></div>
      <div class="row"><span>Breaches from the start to here</span><b>${all[m].hits} of ${all[m].n} (${pct(all[m].rate)}, target ${lvlTxt})</b></div>
      <div class="row"><span>Average miss from the start (lower is better)</span><b>${all[m].loss.toFixed(2)} bp</b></div>
    </div>`;
  $("risk-stats").innerHTML = stat("t", "Timestral", C.timestral) + stat("g", "GARCH (tuned on the previous six months)", C.garch);
  risk.geom = { a, end, X, P };
}

function riskHover(e) {
  const gm = risk.geom;
  if (!gm) return;
  const rect = e.target.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const i = Math.round(gm.a + ((x - gm.P.l) / (gm.P.r - gm.P.l)) * (gm.end - gm.a + 1) - 0.5);
  if (i < gm.a || i > gm.end) { risk.hover = null; tooltip(e.target, null); drawRisk(); return; }
  risk.hover = i;
  drawRisk();
  const d = risk.data, lv = risk.level;
  const t = d["t" + lv][i], g = d["g" + lv][i], c = d.close[i];
  const mark = (line) => (c < line ? " <b>· breached</b>" : "");
  tooltip(e.target,
    `${fmtDate(d.t0 + i * d.step, true)}<br>price ${fmtPrice(c)} (prev. ${fmtPrice(d.last[i])})<br>` +
    `${sw(C.timestral)}Timestral line ${fmtPrice(t)}${mark(t)}<br>${sw(C.garch)}GARCH line ${fmtPrice(g)}${mark(g)}`, x, e.clientY - rect.top);
}

async function setRiskPair(p) {
  risk.pair = p;
  risk.data = await load(`risk-${p}.json`);
  risk.end = risk.data.close.length - 1;
  risk.pos = risk.end;
  $("risk-scrub").value = 1000;
  drawRisk();
}

function riskFrame(ts) {
  if (!risk.playing || !risk.data) return;
  const n = risk.data.close.length;
  const dt = risk.lastTs ? Math.min(0.1, (ts - risk.lastTs) / 1000) : 0;
  risk.lastTs = ts;
  risk.pos = Math.min(n - 1, risk.pos + dt * BASE_HOURS_PER_SECOND * risk.speed);
  const next = Math.floor(risk.pos);
  if (next !== risk.end) {
    risk.end = next;
    $("risk-scrub").value = Math.round((risk.end / (n - 1)) * 1000);
    drawRisk();
  }
  if (risk.end >= n - 1) { risk.playing = false; $("risk-play").textContent = "Play"; return; }
  requestAnimationFrame(riskFrame);
}

// ---------------------------------------------------------------- 2. volatility race

const vol = { data: null, hover: null };

function drawVol() {
  const d = vol.data;
  if (!d) return;
  const n = d.time.length;
  // top: realized vs forecasts
  {
    const canvas = $("vol-chart");
    const { ctx, w, h } = setup(canvas);
    const P = { l: 48, r: w - 92, t: 14, b: h - 26 };
    const series = ["timestral", "garch", "blend"];
    let hi = 0;
    for (let i = 0; i < n; i++) hi = Math.max(hi, d.realized[i], ...series.map((s) => d[s][i]));
    hi *= 1.05;
    const X = (i) => P.l + ((i + 0.5) / n) * (P.r - P.l);
    const Y = (v) => P.b - (v / hi) * (P.b - P.t);
    yAxis(ctx, P, Y, 0, hi, (v) => `${v.toFixed(0)}%`);
    xDates(ctx, P, X, 0, n - 1, (i) => d.time[Math.max(0, Math.min(n - 1, Math.round(i)))]);
    const bw = Math.max(1, ((P.r - P.l) / n) * 0.7);
    ctx.fillStyle = C.realized;
    for (let i = 0; i < n; i++) ctx.fillRect(X(i) - bw / 2, Y(d.realized[i]), bw, P.b - Y(d.realized[i]));
    for (const s of series) {
      ctx.strokeStyle = C[s];
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(d[s][i]));
      ctx.stroke();
    }
    const labels = series.map((s) => [Y(d[s][n - 1]), NAMES[s], C[s]]).concat([[Y(d.realized[n - 1]), "actual", C.text]]).sort((p, q) => p[0] - q[0]);
    let last = -Infinity;
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const [y0, text, color] of labels) {
      const y = Math.max(y0, last + 14);
      last = y;
      ctx.fillStyle = color;
      ctx.fillText(text, P.r + 6, Math.min(y, P.b - 4));
    }
    if (vol.hover != null) {
      ctx.strokeStyle = "rgba(230,237,243,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(vol.hover), P.t); ctx.lineTo(X(vol.hover), P.b); ctx.stroke();
    }
    vol.geomTop = { X, P, n };
  }
  // bottom: running average QLIKE
  {
    const canvas = $("race-chart");
    const { ctx, w, h } = setup(canvas);
    const P = { l: 48, r: w - 120, t: 12, b: h - 22 };
    const series = ["timestral", "garch", "blend"];
    const run = {};
    for (const s of series) {
      let sum = 0;
      run[s] = d.qlike[s].map((v, i) => (sum += v) / (i + 1));
    }
    const start = Math.min(14, n - 1); // the first days are too noisy to compare
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (let i = start; i < n; i++) { lo = Math.min(lo, run[s][i]); hi = Math.max(hi, run[s][i]); }
    const padY = (hi - lo) * 0.1 || 0.05;
    lo -= padY; hi += padY;
    const X = (i) => P.l + ((i + 0.5) / n) * (P.r - P.l);
    const Y = (v) => P.b - ((v - lo) / (hi - lo)) * (P.b - P.t);
    yAxis(ctx, P, Y, lo, hi, (v) => v.toFixed(2));
    ctx.fillStyle = C.muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("running average forecast error (QLIKE), lower is better", P.l + 4, P.t);
    for (const s of series) {
      ctx.strokeStyle = C[s];
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = start; i < n; i++) (i > start ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(run[s][i]));
      ctx.stroke();
    }
    const labels = series.map((s) => [Y(run[s][n - 1]), `${NAMES[s]} ${run[s][n - 1].toFixed(3)}`, C[s]]).sort((p, q) => p[0] - q[0]);
    let last = -Infinity;
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const [y0, text, color] of labels) {
      const y = Math.max(y0, last + 14);
      last = y;
      ctx.fillStyle = color;
      ctx.fillText(text, P.r + 6, Math.min(y, P.b - 4));
    }
    if (vol.hover != null) {
      ctx.strokeStyle = "rgba(230,237,243,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(vol.hover), P.t); ctx.lineTo(X(vol.hover), P.b); ctx.stroke();
    }
    vol.run = run;
  }
}

function volHover(e) {
  const gm = vol.geomTop;
  if (!gm) return;
  const rect = e.target.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const i = Math.round(((x - gm.P.l) / (gm.P.r - gm.P.l)) * gm.n - 0.5);
  if (i < 0 || i >= gm.n) { vol.hover = null; tooltip(e.target, null); drawVol(); return; }
  vol.hover = i;
  drawVol();
  const d = vol.data;
  tooltip(e.target,
    `${fmtDate(d.time[i])}, next 24 h<br>actual volatility ${d.realized[i].toFixed(1)}%<br>` +
    ["timestral", "garch", "blend"].map((s) => `${sw(C[s])}${NAMES[s]} expected ${d[s][i].toFixed(1)}% · running error ${vol.run[s][i].toFixed(3)}`).join("<br>"),
    x, e.clientY - rect.top);
}

async function setVolPair(p) {
  vol.data = await load(`vol-${p}.json`);
  vol.hover = null;
  drawVol();
}

// ---------------------------------------------------------------- headline cards

function cards(idx) {
  const h = idx.headline;
  const v = h.var;
  const items = [
    ["Forecast ranges vs GARCH", signed(h.crps_vs_garch.avg), `CRPS skill, 1 to 24 h ahead, 90% CI ${signed(h.crps_vs_garch.ci[0])} to ${signed(h.crps_vs_garch.ci[1])}`],
    ["Blend vs GARCH", signed(h.blend_vs_garch.avg), `Timestral + GARCH averaged, CI ${signed(h.blend_vs_garch.ci[0])} to ${signed(h.blend_vs_garch.ci[1])}`],
    ["Timestral 1-in-100 line", pct(v.timestral.hit_rate), `crossed (target 1%), rejected on ${v.timestral.kupiec_rejections} of 8 coins`],
    ["GARCH 1-in-20 line", pct(v.garch["5"].hit_rate), `crossed (target 5%): too cautious, rejected on ${v.garch["5"].kupiec_rejections} of 8 coins`],
  ];
  $("cards").innerHTML = items.map(([k, val, s]) => `<div class="card"><div class="k">${k}</div><div class="v">${val}</div><div class="s">${s}</div></div>`).join("");
  $("span-dates").textContent = `${fmtDate(idx.start)} to ${fmtDate(idx.end)}`;
}

// ---------------------------------------------------------------- wiring

(async function init() {
  const idx = await load("index.json");
  cards(idx);
  for (const id of ["risk-pair", "vol-pair"]) {
    $(id).replaceChildren(...idx.pairs.map((p) => Object.assign(document.createElement("option"), { value: p, textContent: p.replace("USDT", "") })));
  }
  $("risk-pair").addEventListener("change", (e) => setRiskPair(e.target.value));
  $("vol-pair").addEventListener("change", (e) => setVolPair(e.target.value));
  $("risk-window").addEventListener("change", (e) => { risk.win = Number(e.target.value); drawRisk(); });
  for (const [id, lv] of [["lvl-01", "01"], ["lvl-05", "05"]]) {
    $(id).addEventListener("click", () => {
      risk.level = lv;
      $("lvl-01").classList.toggle("on", lv === "01");
      $("lvl-05").classList.toggle("on", lv === "05");
      drawRisk();
    });
  }
  $("risk-scrub").addEventListener("input", (e) => {
    const n = risk.data.close.length;
    risk.end = Math.max(1, Math.round((Number(e.target.value) / 1000) * (n - 1)));
    risk.pos = risk.end;
    risk.playing = false;
    $("risk-play").textContent = "Play";
    drawRisk();
  });
  $("risk-play").addEventListener("click", () => {
    risk.playing = !risk.playing;
    $("risk-play").textContent = risk.playing ? "Pause" : "Play";
    if (risk.playing) {
      if (risk.end >= risk.data.close.length - 1) risk.end = Math.min(risk.win, risk.data.close.length - 1);
      risk.pos = risk.end;
      risk.lastTs = 0;
      requestAnimationFrame(riskFrame);
    }
  });
  $("risk-speed").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-speed]");
    if (!b) return;
    risk.speed = Number(b.dataset.speed);
    for (const x of $("risk-speed").querySelectorAll("button")) x.classList.toggle("on", x === b);
  });
  $("risk-chart").addEventListener("mousemove", riskHover);
  $("risk-chart").addEventListener("mouseleave", (e) => { risk.hover = null; tooltip(e.target, null); drawRisk(); });
  for (const id of ["vol-chart", "race-chart"]) {
    $(id).addEventListener("mousemove", volHover);
    $(id).addEventListener("mouseleave", (e) => { vol.hover = null; tooltip(e.target, null); drawVol(); });
  }
  window.addEventListener("resize", () => { drawRisk(); drawVol(); });
  await setRiskPair(idx.pairs[0]);
  await setVolPair(idx.pairs[0]);
})();
