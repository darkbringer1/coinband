// coinband UI: runs a backtest job, then animates the forecast bands over the real price on a canvas.
"use strict";

const $ = (id) => document.getElementById(id);

const COLORS = {
  bg: "#151b23",
  grid: "rgba(139,152,168,0.12)",
  axis: "#8b98a8",
  price: "#e6edf3",
  band80: "rgba(74,163,255,0.30)",
  band90: "rgba(74,163,255,0.14)",
  trail: "74,163,255",
  median: "#f0a83a",
  bad: "#ff5c5c",
  good: "#3ecf8e",
  playhead: "rgba(240,168,58,0.55)",
  text: "#e6edf3",
  muted: "#8b98a8",
};
const L = { P05: 0, P10: 1, P50: 2, P90: 3, P95: 4 };
const FORM_FIELDS = ["symbol", "interval", "model", "bars", "context", "horizon", "stride", "max_origins"];
const STORE_KEY = "coinband.form";
const AUTO_SECONDS = 30;
const HOLD_SECONDS = 2; // pause on the live forecast at the end of a recording

const canvas = $("chart");
const ctx = canvas.getContext("2d");

const state = {
  data: null, // prepared result
  p: 0, // playhead: bars with index < p are known
  playing: false,
  view: "follow",
  trail: true,
  lastTs: 0,
  cam: null, // smoothed {x0, x1, y0, y1}
  recorder: null,
  holdUntil: 0,
  fixedSize: null, // {w, h} while recording
};

// ---------- formatting ----------

function fmtPrice(v) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (a >= 100) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (a >= 1) return v.toLocaleString("en-US", { maximumFractionDigits: 4 });
  return v.toPrecision(4);
}
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : `${(v * 100).toFixed(d)}%`);
const signedPct = (v, d = 2) => (v == null || !isFinite(v) ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);

function fmtTime(ms, intervalMs, withDate = true) {
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, "0");
  const date = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
  if (intervalMs >= 86_400_000) return date;
  const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  return withDate ? `${date.slice(5)} ${hm}` : hm;
}

// ---------- API ----------

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* empty */ }
  if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
  return body;
}

async function loadSymbols() {
  try {
    const { symbols } = await api("/api/symbols");
    const list = $("symbols");
    const frag = document.createDocumentFragment();
    for (const s of symbols) {
      const o = document.createElement("option");
      o.value = s.symbol;
      o.label = `${s.base} / ${s.quote}`;
      frag.appendChild(o);
    }
    list.replaceChildren(frag);
  } catch (e) {
    console.warn("symbol list unavailable:", e.message);
  }
}

async function loadModels() {
  const { models, default: def, device } = await api("/api/models");
  const sel = $("model");
  sel.replaceChildren(
    ...models.map((m) => {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = `${m.name} (${m.loaded ? "loaded" : `${m.size_mb} MB`})`;
      return o;
    }),
  );
  sel.value = def;
  sel.title = `Inference device: ${device}`;
}

function saveForm() {
  const v = {};
  for (const f of FORM_FIELDS) v[f] = $(f).value;
  localStorage.setItem(STORE_KEY, JSON.stringify(v));
}

function restoreForm() {
  try {
    const v = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    for (const f of FORM_FIELDS) {
      if (v[f] == null) continue;
      const el = $(f);
      if (el.tagName === "SELECT" && ![...el.options].some((o) => o.value === v[f])) continue;
      el.value = v[f];
    }
  } catch { /* ignore */ }
}

function setProgress(text, frac) {
  $("progress").hidden = false;
  $("progress-text").textContent = text;
  $("progress-fill").style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
}

function showError(msg) {
  const el = $("error");
  el.hidden = !msg;
  el.textContent = msg || "";
}

async function runBacktest(ev) {
  ev.preventDefault();
  showError("");
  saveForm();
  const body = {};
  for (const f of FORM_FIELDS) body[f] = $(f).value;
  body.symbol = body.symbol.trim().toUpperCase();
  for (const f of ["bars", "context", "horizon", "stride", "max_origins"]) body[f] = Number(body[f]);

  $("run").disabled = true;
  setProgress("Submitting…", 0);
  try {
    const { job_id } = await api("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    for (;;) {
      await new Promise((r) => setTimeout(r, 400));
      const job = await api(`/api/jobs/${job_id}`);
      if (job.status === "error") throw new Error(job.error);
      if (job.status === "done") {
        setProgress(`Done: ${job.result.origins.length} origins in ${job.result.inference_s}s`, 1);
        loadResult(job.result);
        loadModels().then(() => ($("model").value = body.model)).catch(() => {});
        break;
      }
      const frac = job.total ? job.done / job.total : 0;
      const label = {
        fetching: `Fetching candles ${job.done}/${job.total}`,
        "loading model": "Loading model (first run downloads the weights)…",
        "waiting for model": "Waiting for another job…",
        forecasting: `Forecasting ${job.done}/${job.total} origins`,
      }[job.phase] || job.phase;
      setProgress(label, frac);
    }
  } catch (e) {
    showError(e.message);
    $("progress").hidden = true;
  } finally {
    $("run").disabled = false;
  }
}

// ---------- data prep ----------

function loadResult(r) {
  const n = r.close.length;
  const H = r.params.horizon;
  const origins = r.origins.map((o) => ({ ...o, end: o.index + H }));
  // For each realized bar, the most recent origin whose horizon covers it: that band was "current".
  const breach = new Uint8Array(n);
  const governed = new Int32Array(n).fill(-1);
  origins.forEach((o, oi) => {
    for (let k = 0; k < H && o.index + k < n; k++) governed[o.index + k] = oi;
  });
  let breaches = 0;
  for (let t = 0; t < n; t++) {
    const oi = governed[t];
    if (oi < 0) continue;
    const b = origins[oi].bands[t - origins[oi].index];
    if (r.close[t] < b[L.P10] || r.close[t] > b[L.P90]) {
      breach[t] = 1;
      breaches++;
    }
  }
  state.data = {
    ...r,
    n,
    H,
    origins,
    breach,
    breaches,
    start: origins[0].index,
    end: n,
    live: origins[origins.length - 1],
  };
  state.p = state.data.start;
  state.cam = null;
  state.playing = false;
  $("empty").hidden = true;
  for (const id of ["play", "restart", "scrub", "record"]) $(id).disabled = false;
  updateLive();
  play(true);
}

function timeAt(t) {
  const d = state.data;
  if (t < d.n) return d.time[Math.max(0, Math.floor(t))];
  return d.time[d.n - 1] + (Math.floor(t) - d.n + 1) * d.interval_ms;
}

function currentOriginIndex(p) {
  // Last origin with index <= p (binary search).
  const os = state.data.origins;
  let lo = 0, hi = os.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (os[mid].index <= p) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

function speedBarsPerSec() {
  const v = $("speed").value;
  const d = state.data;
  if (v === "auto") return Math.max(1, (d.end - d.start) / AUTO_SECONDS);
  return Number(v);
}

// ---------- metrics ----------

function liveMetrics(p) {
  const d = state.data;
  let cov = 0, mae = 0, naive = 0, hits = 0, scored = 0;
  for (const o of d.origins) {
    if (o.end > p || o.direction_hit == null) continue;
    scored++;
    cov += o.coverage;
    mae += o.mae;
    naive += o.naive_mae;
    hits += o.direction_hit ? 1 : 0;
  }
  let breaches = 0;
  const lim = Math.min(d.n, Math.floor(p));
  for (let t = d.start; t < lim; t++) breaches += d.breach[t];
  return {
    scored,
    coverage: scored ? cov / scored : null,
    mae: scored ? mae / scored : null,
    naive: scored ? naive / scored : null,
    skill: scored && naive > 0 ? 1 - mae / naive : null,
    direction: scored ? hits / scored : null,
    breaches,
  };
}

function updateMetrics(m) {
  const cov = $("m-coverage");
  cov.textContent = pct(m.coverage);
  cov.className = "v" + (m.coverage == null ? "" : Math.abs(m.coverage - 0.8) <= 0.08 ? " good" : " bad");
  const sk = $("m-skill");
  sk.textContent = m.skill == null ? "–" : signedPct(m.skill, 1);
  sk.className = "v" + (m.skill == null ? "" : m.skill > 0 ? " good" : " bad");
  $("m-mae").textContent = m.mae == null ? "naive = price stays flat" : `${fmtPrice(m.mae)} vs naive ${fmtPrice(m.naive)}`;
  const dir = $("m-direction");
  dir.textContent = pct(m.direction);
  dir.className = "v" + (m.direction == null ? "" : m.direction > 0.5 ? " good" : " bad");
  $("m-scored").textContent = m.scored ? String(m.scored) : "–";
  $("m-breaches").textContent = `${m.breaches} bars outside the band`;
}

function updateLive() {
  const d = state.data;
  const live = d.live;
  const last = d.close[d.n - 1];
  const endBand = live.bands[d.H - 1];
  $("m-live").textContent = `${fmtPrice(endBand[L.P50])} (${signedPct(endBand[L.P50] / last - 1)})`;
  $("m-live-band").textContent =
    `80% band ${fmtPrice(endBand[L.P10])} – ${fmtPrice(endBand[L.P90])} at ${fmtTime(timeAt(d.n + d.H - 1), d.interval_ms)} UTC`;
}

// ---------- drawing ----------

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (state.fixedSize) return;
  const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function targetCamera(p) {
  const d = state.data;
  let x0, x1;
  if (state.view === "full") {
    x0 = Math.max(0, d.start - d.H * 2);
    x1 = d.n + d.H;
  } else {
    const width = Math.min(Math.max(d.H * 8, 160), d.n + d.H);
    x1 = p + Math.max(d.H * 1.3, width * 0.3);
    x0 = x1 - width;
  }
  // y range over what is visible: price up to the playhead and the bands drawn.
  let lo = Infinity, hi = -Infinity;
  const a = Math.max(0, Math.floor(x0)), b = Math.min(d.n - 1, Math.ceil(Math.min(x1, p)));
  const step = Math.max(1, Math.floor((b - a) / 2000));
  for (let t = a; t <= b; t += step) {
    lo = Math.min(lo, d.close[t]);
    hi = Math.max(hi, d.close[t]);
  }
  const ci = currentOriginIndex(p);
  const first = state.trail || state.view === "full" ? 0 : Math.max(0, ci);
  for (let i = Math.max(0, first); i <= ci; i++) {
    const o = d.origins[i];
    if (o.end < x0 || o.index > x1) continue;
    const lim = i === ci ? d.H : Math.min(d.H, Math.ceil(p) - o.index);
    for (let k = 0; k < lim; k++) {
      lo = Math.min(lo, o.bands[k][i === ci ? L.P05 : L.P10]);
      hi = Math.max(hi, o.bands[k][i === ci ? L.P95 : L.P90]);
    }
  }
  if (!isFinite(lo)) { lo = d.close[d.n - 1] * 0.95; hi = d.close[d.n - 1] * 1.05; }
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.01 || 1;
  return { x0, x1, y0: lo - pad, y1: hi + pad };
}

function niceStep(range, target) {
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
}

function draw() {
  resize();
  const W = canvas.width, Hh = canvas.height;
  const s = W / 1280; // layout unit: everything is laid out on a 1280-wide design
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, Hh);
  const d = state.data;
  if (!d) return;
  ctx.scale(s, s);
  const w = 1280, h = Hh / s;
  const plot = { l: 16, r: w - 86, t: 78, b: h - 62 };

  const p = state.p;
  const target = targetCamera(p);
  if (!state.cam) state.cam = { ...target };
  else {
    const k = state.recorder || state.playing ? 0.15 : 0.35;
    for (const key of ["x0", "x1", "y0", "y1"]) state.cam[key] += (target[key] - state.cam[key]) * k;
  }
  const { x0, x1, y0, y1 } = state.cam;
  const X = (t) => plot.l + ((t - x0) / (x1 - x0)) * (plot.r - plot.l);
  const Y = (v) => plot.b - ((v - y0) / (y1 - y0)) * (plot.b - plot.t);

  // grid + price axis
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "middle";
  const ystep = niceStep(y1 - y0, 6);
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = COLORS.axis;
  ctx.textAlign = "left";
  for (let v = Math.ceil(y0 / ystep) * ystep; v <= y1; v += ystep) {
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(plot.l, y); ctx.lineTo(plot.r, y); ctx.stroke();
    ctx.fillText(fmtPrice(v), plot.r + 8, y);
  }
  // time axis
  const xstepBars = Math.max(1, niceStep(x1 - x0, 7));
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let t = Math.ceil(x0 / xstepBars) * xstepBars; t <= x1; t += xstepBars) {
    const x = X(t);
    if (x < plot.l + 30 || x > plot.r - 30) continue;
    ctx.beginPath(); ctx.moveTo(x, plot.t); ctx.lineTo(x, plot.b); ctx.stroke();
    ctx.fillText(fmtTime(timeAt(t), d.interval_ms), x, plot.b + 8);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.l, plot.t, plot.r - plot.l, plot.b - plot.t);
  ctx.clip();

  const ci = currentOriginIndex(p);

  // trail of past bands (only their realized part)
  if (state.trail) {
    for (let i = 0; i < ci; i++) {
      const o = d.origins[i];
      if (o.end < x0 || o.index > x1) continue;
      const lim = Math.min(d.H, Math.ceil(p) - o.index);
      if (lim <= 0) continue;
      const age = ci - i;
      const alpha = 0.06 + 0.16 * Math.exp(-age / 6);
      fillBand(o, 0, lim, L.P10, L.P90, `rgba(${COLORS.trail},${alpha})`, X, Y, true);
    }
  }

  // current fan: the whole horizon, ahead of the playhead
  if (ci >= 0) {
    const o = d.origins[ci];
    fillBand(o, 0, d.H, L.P05, L.P95, COLORS.band90, X, Y, true);
    fillBand(o, 0, d.H, L.P10, L.P90, COLORS.band80, X, Y, true);
    ctx.strokeStyle = COLORS.median;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(X(o.index - 1), Y(d.close[o.index - 1]));
    for (let k = 0; k < d.H; k++) ctx.lineTo(X(o.index + k), Y(o.bands[k][L.P50]));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // price up to the playhead, with a partial last segment for smooth motion
  const lastBar = Math.min(d.n - 1, Math.floor(p) - 1);
  const a = Math.max(0, Math.floor(x0) - 1);
  if (lastBar >= a) {
    ctx.strokeStyle = COLORS.price;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const pxPerBar = (plot.r - plot.l) / (x1 - x0);
    if (pxPerBar >= 0.5) {
      ctx.moveTo(X(a), Y(d.close[a]));
      for (let t = a + 1; t <= lastBar; t++) ctx.lineTo(X(t), Y(d.close[t]));
    } else {
      // more bars than pixels: draw a min/max envelope per pixel column
      const per = Math.ceil(1 / pxPerBar);
      let first = true;
      for (let t = a; t <= lastBar; t += per) {
        let mn = Infinity, mx = -Infinity;
        for (let u = t; u < Math.min(t + per, lastBar + 1); u++) { mn = Math.min(mn, d.close[u]); mx = Math.max(mx, d.close[u]); }
        const x = X(t);
        if (first) { ctx.moveTo(x, Y(mx)); first = false; } else ctx.lineTo(x, Y(mx));
        ctx.lineTo(x, Y(mn));
      }
    }
    // partial segment toward the next bar
    const frac = p - Math.floor(p);
    const next = lastBar + 1;
    if (frac > 0 && next < d.n && next === Math.floor(p)) {
      const v = d.close[lastBar] + (d.close[next] - d.close[lastBar]) * frac;
      ctx.lineTo(X(lastBar + frac), Y(v));
    }
    ctx.stroke();

    // breaches: realized bars outside the band that was current for them
    ctx.fillStyle = COLORS.bad;
    const r = Math.max(2, Math.min(4, pxPerBar * 0.45));
    for (let t = Math.max(d.start, a); t <= lastBar; t++) {
      if (!d.breach[t]) continue;
      ctx.beginPath();
      ctx.arc(X(t), Y(d.close[t]), r, 0, Math.PI * 2);
      ctx.fill();
    }
    // head dot
    ctx.fillStyle = COLORS.median;
    ctx.beginPath();
    ctx.arc(X(lastBar), Y(d.close[lastBar]), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // playhead
  const px = X(Math.min(p, d.n) - 1);
  ctx.strokeStyle = COLORS.playhead;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(px, plot.t); ctx.lineTo(px, plot.b); ctx.stroke();
  ctx.setLineDash([]);
  if (p >= d.n) {
    ctx.fillStyle = COLORS.median;
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "top";
    const right = px + 120 < plot.r;
    ctx.textAlign = right ? "left" : "right";
    ctx.fillText("LIVE FORECAST", px + (right ? 8 : -8), plot.t + 8);
  }
  ctx.restore();

  drawHud(w, h, plot);
}

function fillBand(o, k0, k1, loL, hiL, color, X, Y, pinch = false) {
  const d = state.data;
  ctx.fillStyle = color;
  ctx.beginPath();
  const startX = pinch ? X(o.index - 1) : X(o.index + k0);
  const anchor = d.close[o.index - 1];
  if (pinch) ctx.moveTo(startX, Y(anchor));
  else ctx.moveTo(startX, Y(o.bands[k0][hiL]));
  for (let k = k0; k < k1; k++) ctx.lineTo(X(o.index + k), Y(o.bands[k][hiL]));
  for (let k = k1 - 1; k >= k0; k--) ctx.lineTo(X(o.index + k), Y(o.bands[k][loL]));
  ctx.closePath();
  ctx.fill();
}

function drawHud(w, h, plot) {
  const d = state.data;
  const m = liveMetrics(state.p);
  updateMetrics(m);

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillStyle = COLORS.text;
  ctx.font = "700 22px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(`${d.symbol} · ${d.interval}`, 16, 32);
  ctx.fillStyle = COLORS.muted;
  ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
  const modelName = d.model.split("/").pop();
  ctx.fillText(
    `${modelName} · walk-forward, ${d.H}-bar horizon, ${d.params.stride}-bar stride, ${d.params.context}-bar context`,
    16, 54,
  );

  // metric chips, top right
  const chips = [
    ["80% coverage", pct(m.coverage), m.coverage == null ? COLORS.text : Math.abs(m.coverage - 0.8) <= 0.08 ? COLORS.good : COLORS.bad],
    ["MAE vs naive", m.skill == null ? "–" : signedPct(m.skill, 1), m.skill == null ? COLORS.text : m.skill > 0 ? COLORS.good : COLORS.bad],
    ["direction hit", pct(m.direction), m.direction == null ? COLORS.text : m.direction > 0.5 ? COLORS.good : COLORS.bad],
    ["origins", String(m.scored), COLORS.text],
  ];
  let x = w - 16;
  ctx.textAlign = "right";
  for (let i = chips.length - 1; i >= 0; i--) {
    const [k, v, c] = chips[i];
    ctx.font = "700 20px ui-sans-serif, system-ui, sans-serif";
    const vw = ctx.measureText(v).width;
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    const kw = ctx.measureText(k).width;
    const cw = Math.max(vw, kw);
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(k, x, 26);
    ctx.fillStyle = c;
    ctx.font = "700 20px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(v, x, 52);
    x -= cw + 28;
  }

  // legend + footer
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "left";
  let lx = 16;
  const ly = h - 12;
  const item = (draw, label) => {
    draw(lx, ly - 4);
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(label, lx + 22, ly);
    lx += 22 + ctx.measureText(label).width + 18;
  };
  item((x0, y) => { ctx.strokeStyle = COLORS.price; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + 16, y); ctx.stroke(); }, "price");
  item((x0, y) => { ctx.strokeStyle = COLORS.median; ctx.lineWidth = 2; ctx.setLineDash([5, 3]); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + 16, y); ctx.stroke(); ctx.setLineDash([]); }, "median");
  item((x0, y) => { ctx.fillStyle = COLORS.band80; ctx.fillRect(x0, y - 5, 16, 10); }, "80% band");
  item((x0, y) => { ctx.fillStyle = COLORS.band90; ctx.fillRect(x0, y - 5, 16, 10); }, "90% band");
  item((x0, y) => { ctx.fillStyle = COLORS.bad; ctx.beginPath(); ctx.arc(x0 + 8, y, 4, 0, Math.PI * 2); ctx.fill(); }, "outside band");
  ctx.textAlign = "right";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText("coinband · unofficial, not affiliated with Mistral AI or Binance · not financial advice", w - 16, ly);
}

// ---------- playback ----------

function setPlayhead(p) {
  const d = state.data;
  state.p = Math.max(d.start, Math.min(d.end, p));
  $("scrub").value = String(Math.round(((state.p - d.start) / Math.max(1, d.end - d.start)) * 1000));
}

function play(on) {
  state.playing = on;
  if (on && state.data && state.p >= state.data.end) setPlayhead(state.data.start);
  $("play").textContent = on ? "Pause" : "Play";
  state.lastTs = 0;
}

function frame(ts) {
  if (state.data) {
    if (state.playing) {
      const dt = state.lastTs ? Math.min(0.1, (ts - state.lastTs) / 1000) : 0;
      setPlayhead(state.p + dt * speedBarsPerSec());
      if (state.p >= state.data.end) {
        state.playing = false;
        $("play").textContent = "Play";
        if (state.recorder) state.holdUntil = ts + HOLD_SECONDS * 1000;
      }
    }
    if (state.recorder && !state.playing && state.holdUntil && ts >= state.holdUntil) {
      state.holdUntil = 0;
      state.recorder.stop();
    }
  }
  state.lastTs = ts;
  draw();
  requestAnimationFrame(frame);
}

// ---------- recording ----------

function pickMime() {
  for (const m of ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return null;
}

function toggleRecord() {
  if (state.recorder) {
    state.recorder.stop();
    return;
  }
  const mime = pickMime();
  if (!mime || !canvas.captureStream) {
    showError("This browser cannot record a canvas to WebM (try Chrome or Firefox).");
    return;
  }
  state.fixedSize = { w: 1920, h: 1080 };
  canvas.width = 1920;
  canvas.height = 1080;
  state.cam = null;
  const stream = canvas.captureStream(60);
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const d = state.data;
    a.download = `coinband-${d.symbol}-${d.interval}-${d.model.split("/").pop()}.webm`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    state.recorder = null;
    state.fixedSize = null;
    state.holdUntil = 0;
    $("record").textContent = "Record .webm";
    $("record").classList.remove("rec");
    resize();
  };
  state.recorder = rec;
  rec.start(250);
  $("record").textContent = "Stop recording";
  $("record").classList.add("rec");
  setPlayhead(state.data.start);
  play(true);
}

// ---------- wiring ----------

function setView(v) {
  state.view = v;
  $("view-follow").classList.toggle("on", v === "follow");
  $("view-full").classList.toggle("on", v === "full");
}

$("run-form").addEventListener("submit", runBacktest);
$("play").addEventListener("click", () => play(!state.playing));
$("restart").addEventListener("click", () => { setPlayhead(state.data.start); state.cam = null; play(true); });
$("scrub").addEventListener("input", (e) => {
  const d = state.data;
  play(false);
  setPlayhead(d.start + (Number(e.target.value) / 1000) * (d.end - d.start));
});
$("view-follow").addEventListener("click", () => setView("follow"));
$("view-full").addEventListener("click", () => setView("full"));
$("trail").addEventListener("change", (e) => (state.trail = e.target.checked));
$("record").addEventListener("click", toggleRecord);
document.addEventListener("keydown", (e) => {
  if (!state.data || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); play(!state.playing); }
  else if (e.key === "ArrowRight") { play(false); setPlayhead(Math.floor(state.p) + (e.shiftKey ? state.data.H : 1)); }
  else if (e.key === "ArrowLeft") { play(false); setPlayhead(Math.ceil(state.p) - (e.shiftKey ? state.data.H : 1)); }
  else if (e.key === "f") setView(state.view === "follow" ? "full" : "follow");
});
window.addEventListener("resize", () => { state.cam = null; });

(async function init() {
  try {
    await loadModels();
  } catch (e) {
    showError(`Server unreachable: ${e.message}`);
  }
  restoreForm();
  // Shareable links: #symbol=ETHUSDT&interval=4h&horizon=12&run fills the form (and runs it).
  const hash = new URLSearchParams(location.hash.slice(1));
  for (const f of FORM_FIELDS) if (hash.has(f)) $(f).value = hash.get(f);
  if (hash.get("view") === "full") setView("full");
  loadSymbols();
  requestAnimationFrame(frame);
  if (hash.has("run")) $("run-form").requestSubmit();
})();
