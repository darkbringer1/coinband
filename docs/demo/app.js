// coinband UI: runs a backtest job, then animates the forecast bands over the real price on a canvas.
"use strict";

const $ = (id) => document.getElementById(id);

const COLORS = {
  bg: "#151b23",
  grid: "rgba(139,152,168,0.12)",
  axis: "#8b98a8",
  price: "#e6edf3",
  future: "rgba(230,237,243,0.38)",
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
const FORM_FIELDS = ["source", "symbol", "interval", "model", "bars", "context", "horizon", "stride", "max_origins", "trend_window"];
const NUMERIC_FIELDS = ["bars", "context", "horizon", "stride", "max_origins", "trend_window"];
const SERIES = ["#3987e5", "#d95926"]; // analysis charts: validated on the dark surface
const STORE_KEY = "coinband.form.v5";
const LIVE_EVERY_MS = 60_000;
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

// Times are shown in the market's own time zone (UTC for crypto).
const fmtCache = new Map();
function fmtParts(ms, tz) {
  let f = fmtCache.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
    fmtCache.set(tz, f);
  }
  const p = Object.fromEntries(f.formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
  return p;
}
function marketTz() { return state.data?.info?.timezone || "UTC"; }
function fmtTime(ms, intervalMs, withDate = true) {
  const p = fmtParts(ms, marketTz());
  const date = `${p.year}-${p.month}-${p.day}`;
  if (intervalMs >= 86_400_000) return date;
  const hm = `${p.hour}:${p.minute}`;
  return withDate ? `${date.slice(5)} ${hm}` : hm;
}
function tzLabel() {
  const tz = marketTz();
  return tz === "UTC" ? "UTC" : tz.split("/").pop().replace("_", " ") + " time";
}

// ---------- API ----------

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* empty */ }
  if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
  return body;
}

let MARKETS = {};
const symbolCache = {};

async function loadSymbols(source) {
  try {
    if (!symbolCache[source]) symbolCache[source] = (await api(`/api/symbols?source=${source}`)).symbols;
    if ($("source").value !== source) return;
    const frag = document.createDocumentFragment();
    for (const s of symbolCache[source]) {
      const o = document.createElement("option");
      o.value = s.symbol;
      o.label = s.base ? `${s.base} / ${s.quote}` : `${s.name} · ${s.group}`;
      frag.appendChild(o);
    }
    $("symbols").replaceChildren(frag);
  } catch (e) {
    console.warn("symbol list unavailable:", e.message);
  }
}

// Intervals, symbol list and leaderboard defaults follow the market; `keep` keeps valid current values.
function applySource(keep = true) {
  const source = $("source").value;
  const m = MARKETS[source];
  if (!m) return;
  const sel = $("interval");
  const prev = sel.value;
  sel.replaceChildren(...m.intervals.map((iv) => Object.assign(document.createElement("option"), { value: iv, textContent: iv })));
  sel.value = keep && m.intervals.includes(prev) ? prev : m.default_interval;
  const sym = $("symbol");
  const looksCrypto = /^[A-Z0-9]+(USDT|USDC|BTC|ETH|BNB|FDUSD|TRY|EUR)$/.test(sym.value);
  if (!keep || (source === "yahoo" && looksCrypto) || (source === "binance" && /[.^=]/.test(sym.value))) sym.value = m.default_symbol;
  if (!keep || !$("lb-symbols").dataset.touched) $("lb-symbols").value = m.leaderboard.join(" ");
  loadSymbols(source);
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
  v.covariates = $("covariates").checked;
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
      if (f === "source") applySource(true);
    }
    if (typeof v.covariates === "boolean") $("covariates").checked = v.covariates;
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

function formBody() {
  const body = {};
  for (const f of FORM_FIELDS) body[f] = $(f).value;
  body.symbol = body.symbol.trim().toUpperCase();
  for (const f of NUMERIC_FIELDS) body[f] = Number(body[f]);
  body.covariates = $("covariates").checked;
  return body;
}

function phaseLabel(job) {
  const m = /^(.*: )?(.*)$/.exec(job.phase);
  const prefix = m[1] || "";
  const label = {
    fetching: `Fetching candles ${job.done}/${job.total}`,
    "loading model": "Loading model (first run downloads the weights)…",
    "waiting for model": "Waiting for another job…",
    forecasting: `Forecasting ${job.done}/${job.total} origins`,
  }[m[2]] || m[2];
  return prefix + label;
}

// Submit a job and poll it to completion; `onProgress(job)` sees each status.
const myJobs = { backtest: null, leaderboard: null };

class Cancelled extends Error {}

async function cancelJob(jobId) {
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  } catch (e) {
    console.warn("cancel failed:", e.message);
  }
  refreshQueue();
}

async function runJob(path, body, onProgress, kind) {
  const { job_id } = await api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (kind) myJobs[kind] = job_id;
  refreshQueue();
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 400));
      const job = await api(`/api/jobs/${job_id}`);
      if (job.status === "error") throw new Error(job.error);
      if (job.status === "cancelled") throw new Cancelled("Cancelled");
      if (job.status === "done") return job.result;
      onProgress(job);
    }
  } finally {
    if (kind && myJobs[kind] === job_id) myJobs[kind] = null;
    refreshQueue();
  }
}

// ---------- queue ----------

async function refreshQueue() {
  if (window.COINBAND_STATIC) return;
  let jobs = [];
  try {
    jobs = (await api("/api/jobs")).jobs;
  } catch {
    return;
  }
  const active = jobs.filter((j) => ["queued", "running", "cancelling"].includes(j.status));
  const recent = jobs.filter((j) => !active.includes(j)).slice(0, 3);
  const shown = active.concat(recent);
  $("queue").hidden = shown.length === 0;
  $("queue-list").replaceChildren(
    ...shown.map((j) => {
      const li = document.createElement("li");
      const isActive = active.includes(j);
      const frac = j.total ? j.done / j.total : 0;
      const state = j.status === "cancelling" ? "cancelling…"
        : !isActive ? j.status
        : j.position > 0 ? `waiting · #${j.position} in line`
        : phaseLabel(j);
      li.innerHTML = `<span class="q-label"></span><span class="q-state"></span>` +
        (isActive ? `<span class="q-bar"><div style="width:${Math.round(frac * 100)}%"></div></span>` : "");
      li.querySelector(".q-label").textContent = j.label + (Object.values(myJobs).includes(j.job_id) ? " (this tab)" : "");
      const st = li.querySelector(".q-state");
      st.textContent = state;
      if (j.status === "error" || j.status === "cancelled") st.classList.add("bad");
      if (isActive && j.status !== "cancelling") {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = "Cancel";
        b.onclick = () => cancelJob(j.job_id);
        li.appendChild(b);
      }
      return li;
    }),
  );
}

async function runBacktest(ev) {
  if (ev) ev.preventDefault();
  showError("");
  saveForm();
  const body = formBody();
  if (myJobs.backtest) cancelJob(myJobs.backtest); // a new run replaces this tab's previous one
  $("cancel-run").hidden = false;
  setProgress("Submitting…", 0);
  try {
    const result = await runJob("/api/backtest", body, (job) =>
      setProgress(job.status === "cancelling" ? "Cancelling…" : phaseLabel(job), job.total ? job.done / job.total : 0),
    "backtest");
    setProgress(`Done: ${result.origins.length} origins in ${result.inference_s}s`, 1);
    loadResult(result);
    loadModels().then(() => ($("model").value = body.model)).catch(() => {});
  } catch (e) {
    if (e instanceof Cancelled) {
      if (!myJobs.backtest) setProgress("Cancelled", 0);
    } else {
      showError(e.message);
      $("progress").hidden = true;
    }
  } finally {
    if (!myJobs.backtest) $("cancel-run").hidden = true;
  }
}

// ---------- data prep ----------

function loadResult(r, opts = {}) {
  state.result = r;
  if (!opts.live) { state.livePrice = null; stopLive(); }
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
  const ribLo = new Float64Array(n).fill(NaN);
  const ribHi = new Float64Array(n).fill(NaN);
  for (let t = 0; t < n; t++) {
    const oi = governed[t];
    if (oi < 0) continue;
    const b = origins[oi].bands[t - origins[oi].index];
    ribLo[t] = b[L.P10];
    ribHi[t] = b[L.P90];
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
    ribLo,
    ribHi,
    start: origins[0].index,
    end: n,
    live: origins[origins.length - 1],
  };
  state.p = state.data.start;
  state.cam = null;
  state.playing = false;
  $("empty").hidden = true;
  for (const id of ["play", "restart", "scrub", "record", "live"]) $(id).disabled = false;
  updateLive();
  renderAnalysis(state.data);
  if (opts.live) {
    setPlayhead(state.data.end);
    play(false);
  } else {
    play(true);
  }
}

function timeAt(t) {
  const d = state.data;
  if (t < d.n) return d.time[Math.max(0, Math.floor(t))];
  const k = Math.floor(t) - d.n;
  const ft = d.future_time || [];
  if (k < ft.length) return ft[k];
  return (ft.length ? ft[ft.length - 1] : d.time[d.n - 1]) + (k - ft.length + 1) * d.interval_ms;
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

// The cone at playhead p: the forecast made at the current origin, anchored on the price head.
// With a forecast every bar, it morphs between consecutive origins so it moves and breathes smoothly.
function coneAt(p) {
  const d = state.data;
  const ci = currentOriginIndex(p);
  if (ci < 0) return null;
  const o = d.origins[ci];
  const next = d.origins[ci + 1];
  const frac = p - o.index;
  const H = d.H;
  const xs = new Array(H);
  let bands = o.bands;
  let ax = o.index - 1, ay = d.close[o.index - 1];
  if (next && next.index - o.index === 1 && frac > 0) {
    bands = o.bands.map((step, k) => step.map((v, j) => v + (next.bands[k][j] - v) * frac));
    ax += frac;
    ay += (d.close[o.index] - ay) * frac;
    for (let k = 0; k < H; k++) xs[k] = o.index + k + frac;
  } else {
    for (let k = 0; k < H; k++) xs[k] = o.index + k;
  }
  return { ci, ax, ay, xs, bands };
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
  let cov = 0, mae = 0, naive = 0, crps = 0, naiveCrps = 0, hits = 0, scored = 0;
  for (const o of d.origins) {
    if (o.end > p || o.direction_hit == null) continue;
    scored++;
    cov += o.coverage;
    crps += o.crps;
    naiveCrps += o.naive_crps;
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
    crpsSkill: scored && naiveCrps > 0 ? 1 - crps / naiveCrps : null,
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
  const cr = $("m-crps");
  cr.textContent = m.crpsSkill == null ? "–" : signedPct(m.crpsSkill, 1);
  cr.className = "v" + (m.crpsSkill == null ? "" : m.crpsSkill > 0 ? " good" : " bad");
  const plain = state.data.summary.plain;
  $("m-crps-sub").textContent = plain && plain.crps_skill != null
    ? `without trend: ${signedPct(plain.crps_skill, 1)} (full run)`
    : "whole band vs naive band";
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
    `80% band ${fmtPrice(endBand[L.P10])} – ${fmtPrice(endBand[L.P90])} at ${fmtTime(timeAt(d.n + d.H - 1), d.interval_ms)} ${tzLabel()}`;
  const lp = state.livePrice;
  if (lp && lp.price != null) {
    // compare with the band forecast for the close of the bar that is forming now (first step)
    const b0 = d.live.bands[0];
    const where = lp.price < b0[L.P10] ? " · below this bar's 80% band" : lp.price > b0[L.P90] ? " · above this bar's 80% band" : " · inside this bar's 80% band";
    $("m-live-band").textContent += ` · now ${fmtPrice(lp.price)}${where}`;
  }
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
  const a = Math.max(0, Math.floor(x0)), b = Math.min(d.n - 1, Math.ceil(x1));
  const step = Math.max(1, Math.floor((b - a) / 2000));
  for (let t = a; t <= b; t += step) {
    lo = Math.min(lo, d.close[t]);
    hi = Math.max(hi, d.close[t]);
  }
  if (state.trail) {
    for (let t = Math.max(a, d.start); t <= b; t += step) {
      if (isNaN(d.ribLo[t])) continue;
      lo = Math.min(lo, d.ribLo[t]);
      hi = Math.max(hi, d.ribHi[t]);
    }
  }
  const cone = coneAt(p);
  if (cone) {
    for (const s of cone.bands) {
      lo = Math.min(lo, s[L.P05]);
      hi = Math.max(hi, s[L.P95]);
    }
  }
  if (state.livePrice?.price != null && p >= d.end - 1e-6) {
    lo = Math.min(lo, state.livePrice.price);
    hi = Math.max(hi, state.livePrice.price);
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

  const pxPerBar = (plot.r - plot.l) / (x1 - x0);

  // trail: the 80% band each realized bar was given by the forecast that was current for it
  if (state.trail) drawRibbon(Math.max(d.start, Math.floor(x0) - 1), Math.min(d.n - 1, Math.floor(p) - 1), pxPerBar, X, Y);

  // the cone: current forecast, anchored on the price head
  const cone = coneAt(p);
  if (cone) drawCone(cone, X, Y);

  // the real price over the whole view: bright behind the playhead, dimmer ahead of it
  const lastBar = Math.min(d.n - 1, Math.floor(p) - 1);
  const a = Math.max(0, Math.floor(x0) - 1);
  const z = Math.min(d.n - 1, Math.ceil(x1) + 1);
  const frac = p - Math.floor(p);
  let headX = lastBar, headY = d.close[lastBar];
  if (frac > 0 && lastBar + 1 < d.n) {
    headX = lastBar + frac;
    headY = d.close[lastBar] + (d.close[lastBar + 1] - d.close[lastBar]) * frac;
  }
  ctx.lineJoin = "round";
  ctx.lineWidth = 1.6;
  if (headX < z) {
    ctx.strokeStyle = COLORS.future;
    ctx.beginPath();
    ctx.moveTo(X(Math.max(headX, a)), Y(headX >= a ? headY : d.close[a]));
    traceSeries(Math.max(lastBar + 1, a), z, pxPerBar, X, Y);
    ctx.stroke();
  }
  if (lastBar >= a) {
    ctx.strokeStyle = COLORS.price;
    ctx.beginPath();
    ctx.moveTo(X(a), Y(d.close[a]));
    traceSeries(a + 1, lastBar, pxPerBar, X, Y);
    ctx.lineTo(X(headX), Y(headY));
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
  }

  // the real future against the cone: ring each bar ahead that the price leaves the cone at
  if (cone) {
    ctx.strokeStyle = COLORS.bad;
    ctx.lineWidth = 1.5;
    const r = Math.max(2.5, Math.min(4.5, pxPerBar * 0.5));
    for (let t = lastBar + 1; t < d.n; t++) {
      const band = coneBandAt(cone, t);
      if (!band) break;
      if (d.close[t] >= band[0] && d.close[t] <= band[1]) continue;
      ctx.beginPath();
      ctx.arc(X(t), Y(d.close[t]), r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  if (lastBar >= 0) {
    ctx.fillStyle = COLORS.median;
    ctx.beginPath();
    ctx.arc(X(headX), Y(headY), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // live: the current price inside the forming bar, against the cone
  const lx = livePriceX();
  if (lx != null && state.livePrice.price != null && p >= d.end - 1e-6) {
    const lp = state.livePrice.price;
    const pulse = 5 + 3 * (0.5 + 0.5 * Math.sin(performance.now() / 300));
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(lx), Y(lp), pulse, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = COLORS.text;
    ctx.beginPath(); ctx.arc(X(lx), Y(lp), 3, 0, Math.PI * 2); ctx.fill();
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(`now ${fmtPrice(lp)}`, X(lx) + 10, Y(lp) - 6);
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

// Append close[t0..t1] to the current path; past one bar per pixel, a min/max envelope per column.
function traceSeries(t0, t1, pxPerBar, X, Y) {
  const c = state.data.close;
  if (pxPerBar >= 0.5) {
    for (let t = t0; t <= t1; t++) ctx.lineTo(X(t), Y(c[t]));
    return;
  }
  const per = Math.ceil(1 / pxPerBar);
  for (let t = t0; t <= t1; t += per) {
    let mn = Infinity, mx = -Infinity;
    for (let u = t; u <= Math.min(t + per - 1, t1); u++) { mn = Math.min(mn, c[u]); mx = Math.max(mx, c[u]); }
    ctx.lineTo(X(t), Y(mx));
    ctx.lineTo(X(t), Y(mn));
  }
}

// The cone's 80% band [P10, P90] at bar t, interpolated along its drawn outline; null past the tip.
function coneBandAt({ ax, ay, xs, bands }, t) {
  if (t <= ax || t > xs[xs.length - 1]) return null;
  let k = 0;
  while (xs[k] < t) k++;
  const x0 = k === 0 ? ax : xs[k - 1];
  const lo0 = k === 0 ? ay : bands[k - 1][L.P10];
  const hi0 = k === 0 ? ay : bands[k - 1][L.P90];
  const w = xs[k] === x0 ? 1 : (t - x0) / (xs[k] - x0);
  return [lo0 + (bands[k][L.P10] - lo0) * w, hi0 + (bands[k][L.P90] - hi0) * w];
}

function drawRibbon(a, b, pxPerBar, X, Y) {
  const d = state.data;
  if (b < a) return;
  const step = pxPerBar >= 0.5 ? 1 : Math.ceil(1 / pxPerBar);
  ctx.fillStyle = `rgba(${COLORS.trail},0.16)`;
  // one polygon per run of bars that have a band
  let t = a;
  while (t <= b) {
    while (t <= b && isNaN(d.ribLo[t])) t++;
    const run = [];
    while (t <= b && !isNaN(d.ribLo[t])) { run.push(t); t += step; }
    if (run.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(X(run[0]), Y(d.ribHi[run[0]]));
    for (const u of run) ctx.lineTo(X(u), Y(d.ribHi[u]));
    for (let i = run.length - 1; i >= 0; i--) ctx.lineTo(X(run[i]), Y(d.ribLo[run[i]]));
    ctx.closePath();
    ctx.fill();
  }
}

function drawCone({ ax, ay, xs, bands }, X, Y) {
  const fill = (loL, hiL, color) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(X(ax), Y(ay));
    for (let k = 0; k < xs.length; k++) ctx.lineTo(X(xs[k]), Y(bands[k][hiL]));
    for (let k = xs.length - 1; k >= 0; k--) ctx.lineTo(X(xs[k]), Y(bands[k][loL]));
    ctx.closePath();
    ctx.fill();
  };
  fill(L.P05, L.P95, COLORS.band90);
  fill(L.P10, L.P90, COLORS.band80);
  // crisp edges on the 80% band make the widening and narrowing easy to follow
  ctx.strokeStyle = "rgba(74,163,255,0.75)";
  ctx.lineWidth = 1;
  for (const lvl of [L.P10, L.P90]) {
    ctx.beginPath();
    ctx.moveTo(X(ax), Y(ay));
    for (let k = 0; k < xs.length; k++) ctx.lineTo(X(xs[k]), Y(bands[k][lvl]));
    ctx.stroke();
  }
  ctx.strokeStyle = COLORS.median;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(X(ax), Y(ay));
  for (let k = 0; k < xs.length; k++) ctx.lineTo(X(xs[k]), Y(bands[k][L.P50]));
  ctx.stroke();
  ctx.setLineDash([]);
  // half-width of the 80% band at the tip, as % of the current price
  const tip = bands[xs.length - 1];
  const half = (tip[L.P90] - tip[L.P10]) / 2 / ay;
  const tx = X(xs[xs.length - 1]) + 8;
  ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(74,163,255,0.95)";
  ctx.fillText(`±${(half * 100).toFixed(2)}%`, tx, Y(tip[L.P50]));
}

function drawHud(w, h, plot) {
  const d = state.data;
  const m = liveMetrics(state.p);
  updateMetrics(m);

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillStyle = COLORS.text;
  ctx.font = "700 22px ui-sans-serif, system-ui, sans-serif";
  const name = d.info?.name && d.info.name !== d.symbol ? `  ${d.info.name}` : "";
  ctx.fillText(`${d.symbol} · ${d.interval}${d.info?.currency ? ` · ${d.info.currency}` : ""}`, 16, 32);
  if (name) {
    const w0 = ctx.measureText(`${d.symbol} · ${d.interval}${d.info?.currency ? ` · ${d.info.currency}` : ""}`).width;
    ctx.font = "15px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(name.length > 40 ? name.slice(0, 39) + "…" : name, 16 + w0, 32);
  }
  ctx.fillStyle = COLORS.muted;
  ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
  const modelName = d.model.split("/").pop();
  ctx.fillText(
    `${modelName}${d.covariates ? " + volume covariates" : ""}${d.trend ? ` + ${d.params.trend_window}-bar trend drift` : ""} · walk-forward, ${d.H}-bar horizon, ${d.params.stride === 1 ? "re-forecast every bar" : `${d.params.stride}-bar stride`}, ${d.params.context}-bar context`,
    16, 54,
  );

  // metric chips, top right
  const chips = [
    ["80% coverage", pct(m.coverage), m.coverage == null ? COLORS.text : Math.abs(m.coverage - 0.8) <= 0.08 ? COLORS.good : COLORS.bad],
    ["MAE vs naive", m.skill == null ? "–" : signedPct(m.skill, 1), m.skill == null ? COLORS.text : m.skill > 0 ? COLORS.good : COLORS.bad],
    ["CRPS skill", m.crpsSkill == null ? "–" : signedPct(m.crpsSkill, 1), m.crpsSkill == null ? COLORS.text : m.crpsSkill > 0 ? COLORS.good : COLORS.bad],
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
  item((x0, y) => { ctx.strokeStyle = COLORS.future; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + 16, y); ctx.stroke(); }, "price ahead");
  item((x0, y) => { ctx.fillStyle = COLORS.bad; ctx.beginPath(); ctx.arc(x0 + 8, y, 4, 0, Math.PI * 2); ctx.fill(); }, "left its band");
  item((x0, y) => { ctx.strokeStyle = COLORS.bad; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x0 + 8, y, 4, 0, Math.PI * 2); ctx.stroke(); }, "leaves the cone");
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

// ---------- analysis charts ----------

function fmtSigned(v, d = 1) { return v == null ? "–" : signedPct(v, d); }

// A small single-axis line/dot chart on a canvas, with a crosshair tooltip.
// spec: {series: [{name, color, points: [[x, y]], dots?}], x: [min, max], y: [min, max], xLabel, yFmt, xFmt,
//        refs: [{y, label}], diagonal?, tip: (i) => html}
function smallChart(canvas, spec) {
  const box = canvas.parentElement;
  const tip = box.querySelector(".tip");
  const state = { hover: null };
  const render = () => {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    const P = { l: 44, r: w - 12, t: 10, b: h - 38 };
    const [x0, x1] = spec.x, [y0, y1] = spec.y;
    const X = (v) => P.l + ((v - x0) / (x1 - x0 || 1)) * (P.r - P.l);
    const Y = (v) => P.b - ((v - y0) / (y1 - y0 || 1)) * (P.b - P.t);
    spec._geom = { X, Y, P };
    c.font = "11px ui-sans-serif, system-ui, sans-serif";
    c.fillStyle = COLORS.muted;
    c.strokeStyle = COLORS.grid;
    c.lineWidth = 1;
    // y grid
    const ys = niceStep(y1 - y0, 4);
    c.textAlign = "right"; c.textBaseline = "middle";
    for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-9; v += ys) {
      c.beginPath(); c.moveTo(P.l, Y(v)); c.lineTo(P.r, Y(v)); c.stroke();
      c.fillText(spec.yFmt(v), P.l - 6, Y(v));
    }
    // x ticks
    const xs = Math.max(spec.xMinStep || 0, niceStep(x1 - x0, 5));
    c.textAlign = "center"; c.textBaseline = "top";
    for (let v = Math.ceil(x0 / xs) * xs; v <= x1 + 1e-9; v += xs) c.fillText(spec.xFmt(v), X(v), P.b + 6);
    c.textAlign = "right"; c.textBaseline = "bottom";
    c.fillText(spec.xLabel, P.r, h - 1);
    // references
    c.setLineDash([4, 4]);
    c.strokeStyle = "rgba(139,152,168,0.6)";
    for (const ref of spec.refs || []) {
      c.beginPath(); c.moveTo(P.l, Y(ref.y)); c.lineTo(P.r, Y(ref.y)); c.stroke();
      c.textAlign = "left"; c.textBaseline = "bottom";
      c.fillText(ref.label, P.l + 4, Y(ref.y) - 2);
    }
    if (spec.diagonal) {
      c.beginPath(); c.moveTo(X(x0), Y(x0)); c.lineTo(X(x1), Y(x1)); c.stroke();
    }
    c.setLineDash([]);
    // series
    for (const s of spec.series) {
      c.strokeStyle = s.color;
      c.lineWidth = 2;
      c.lineJoin = "round";
      c.beginPath();
      let started = false;
      for (const [x, y] of s.points) {
        if (y == null) { started = false; continue; }
        if (!started) { c.moveTo(X(x), Y(y)); started = true; } else c.lineTo(X(x), Y(y));
      }
      c.stroke();
      if (s.dots) {
        for (const [x, y] of s.points) {
          if (y == null) continue;
          c.fillStyle = s.color;
          c.strokeStyle = COLORS.bg;
          c.lineWidth = 2;
          c.beginPath(); c.arc(X(x), Y(y), 4, 0, Math.PI * 2); c.fill(); c.stroke();
        }
      }
    }
    // direct labels at the line ends (<= 4 series)
    if (spec.series.length > 1) {
      c.font = "11px ui-sans-serif, system-ui, sans-serif";
      c.textAlign = "right"; c.textBaseline = "bottom";
      for (const s of spec.series) {
        const last = [...s.points].reverse().find((p) => p[1] != null);
        if (!last) continue;
        c.fillStyle = COLORS.text;
        c.fillText(s.name, X(last[0]) - 4, Y(last[1]) - 6);
      }
    }
    // crosshair
    if (state.hover != null) {
      const px = spec.series[0].points[state.hover][0];
      c.strokeStyle = "rgba(230,237,243,0.35)";
      c.lineWidth = 1;
      c.beginPath(); c.moveTo(X(px), P.t); c.lineTo(X(px), P.b); c.stroke();
      for (const s of spec.series) {
        const y = s.points[state.hover][1];
        if (y == null) continue;
        c.fillStyle = s.color; c.strokeStyle = COLORS.bg; c.lineWidth = 2;
        c.beginPath(); c.arc(X(px), Y(y), 4.5, 0, Math.PI * 2); c.fill(); c.stroke();
      }
    }
  };
  canvas.onmousemove = (e) => {
    const g = spec._geom;
    if (!g) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const pts = spec.series[0].points;
    let best = 0;
    for (let i = 1; i < pts.length; i++) if (Math.abs(g.X(pts[i][0]) - mx) < Math.abs(g.X(pts[best][0]) - mx)) best = i;
    state.hover = best;
    render();
    tip.hidden = false;
    tip.innerHTML = spec.tip(best);
    const tx = g.X(pts[best][0]);
    const left = tx + 12 + tip.offsetWidth > rect.width ? tx - 12 - tip.offsetWidth : tx + 12;
    tip.style.left = `${left}px`;
    tip.style.top = `${g.P.t + 4}px`;
  };
  canvas.onmouseleave = () => { state.hover = null; tip.hidden = true; render(); };
  render();
  return render;
}

const analysisCharts = [];

function renderAnalysis(d) {
  const s = d.summary;
  const sec = $("analysis");
  if (!s || !s.scored) { sec.hidden = true; return; }
  sec.hidden = false;
  const H = d.H;
  const steps = Array.from({ length: H }, (_, k) => k + 1);
  const bh = s.by_horizon;
  const sw = (color) => `<span class="sw" style="background:${color}"></span>`;
  const plain = s.plain;
  $("analysis-sub").textContent =
    `${s.scored} fully played-out forecasts · calibration error ${pct(s.calibration_error, 1)}` +
    (plain ? ` · CRPS skill ${fmtSigned(s.crps_skill)} with trend vs ${fmtSigned(plain.crps_skill)} without` : "");

  const covVals = bh.coverage;
  const cmin = Math.min(0.5, ...covVals) - 0.02, cmax = Math.max(1, ...covVals);
  const skills = [...bh.mae_skill, ...bh.crps_skill].filter((v) => v != null);
  const sLo = Math.min(-0.05, ...skills), sHi = Math.max(0.05, ...skills);
  const pad = (sHi - sLo) * 0.1;
  const cal = s.calibration;

  analysisCharts.length = 0;
  analysisCharts.push(smallChart($("c-coverage"), {
    series: [{ name: "coverage", color: SERIES[0], points: steps.map((k, i) => [k, covVals[i]]) }],
    x: [1, H], y: [cmin, cmax], xMinStep: 1, xLabel: "bars ahead", xFmt: (v) => String(v), yFmt: (v) => pct(v, 0),
    refs: [{ y: 0.8, label: "80% target" }],
    tip: (i) => `${i + 1} bar${i ? "s" : ""} ahead<br>${sw(SERIES[0])}coverage ${pct(covVals[i])}`,
  }));
  analysisCharts.push(smallChart($("c-skill"), {
    series: [
      { name: "MAE skill", color: SERIES[0], points: steps.map((k, i) => [k, bh.mae_skill[i]]) },
      { name: "CRPS skill", color: SERIES[1], points: steps.map((k, i) => [k, bh.crps_skill[i]]) },
    ],
    x: [1, H], y: [sLo - pad, sHi + pad], xMinStep: 1, xLabel: "bars ahead", xFmt: (v) => String(v), yFmt: (v) => fmtSigned(v, 0),
    refs: [{ y: 0, label: "naive" }],
    tip: (i) => `${i + 1} bar${i ? "s" : ""} ahead<br>${sw(SERIES[0])}MAE skill ${fmtSigned(bh.mae_skill[i])}<br>${sw(SERIES[1])}CRPS skill ${fmtSigned(bh.crps_skill[i])}`,
  }));
  analysisCharts.push(smallChart($("c-calibration"), {
    series: [{ name: "observed", color: SERIES[0], dots: true, points: cal.levels.map((l, i) => [l, cal.observed[i]]) }],
    x: [0, 1], y: [0, 1], xLabel: "quantile level", xFmt: (v) => pct(v, 0), yFmt: (v) => pct(v, 0), diagonal: true,
    tip: (i) => `P${Math.round(cal.levels[i] * 100)}<br>${sw(SERIES[0])}price at or below it ${pct(cal.observed[i])} of the time<br>calibrated: ${pct(cal.levels[i], 0)}`,
  }));

  const rows = steps.map((k, i) => `<tr><td>${k}</td><td>${pct(covVals[i])}</td><td>${fmtSigned(bh.mae_skill[i])}</td><td>${fmtSigned(bh.crps_skill[i])}</td></tr>`).join("");
  const calRows = cal.levels.map((l, i) => `<tr><td>P${Math.round(l * 100)}</td><td>${pct(cal.observed[i])}</td></tr>`).join("");
  $("horizon-table").innerHTML =
    `<table><thead><tr><th>Bars ahead</th><th>80% coverage</th><th>MAE skill</th><th>CRPS skill</th></tr></thead><tbody>${rows}</tbody></table>` +
    `<table style="margin-top:12px"><thead><tr><th>Quantile</th><th>Price at or below it</th></tr></thead><tbody>${calRows}</tbody></table>`;
}

window.addEventListener("resize", () => analysisCharts.forEach((r) => r()));

// ---------- leaderboard ----------

const lb = { rows: [], sort: "crps_skill", trend: false };
const LB_COLS = [
  ["symbol", "Pair", (r) => r.symbol],
  ["crps_skill", "CRPS skill", (r) => r.summary?.crps_skill, fmtSigned],
  ["skill", "MAE skill", (r) => r.summary?.skill, fmtSigned],
  ["coverage", "80% coverage", (r) => r.summary?.coverage, (v) => pct(v)],
  ["calibration_error", "Calib. error", (r) => r.summary?.calibration_error, (v) => pct(v)],
  ["direction_hit_rate", "Direction hit", (r) => r.summary?.direction_hit_rate, (v) => pct(v)],
  ["plain_crps_skill", "CRPS skill, no trend", (r) => r.plain_crps_skill, fmtSigned],
  ["live_change", "Live median", (r) => r.live_change, (v) => fmtSigned(v, 2)],
];

function renderLeaderboard() {
  const cols = LB_COLS.filter((c) => c[0] !== "plain_crps_skill" || lb.trend);
  const col = cols.find((c) => c[0] === lb.sort) || cols[1];
  // lower is better only for calibration error; errors sink to the bottom
  const dir = col[0] === "calibration_error" || col[0] === "symbol" ? 1 : -1;
  const rows = [...lb.rows].sort((a, b) => {
    const va = a.error ? null : col[2](a), vb = b.error ? null : col[2](b);
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
  });
  const head = cols.map((c) => `<th data-k="${c[0]}" class="${c[0] === col[0] ? "sorted" : ""}">${c[1]}</th>`).join("");
  const body = rows.map((r) => {
    if (r.error) return `<tr data-s="${r.symbol}"><td>${r.symbol}</td><td colspan="${cols.length - 1}" class="bad" style="text-align:left">${r.error}</td></tr>`;
    return `<tr data-s="${r.symbol}">` + cols.map((c, i) => {
      const v = c[2](r);
      if (i === 0) return `<td>${v}</td>`;
      const good = c[0] === "calibration_error" ? null : c[0] === "coverage" ? Math.abs(v - 0.8) <= 0.08 : c[0] === "direction_hit_rate" ? v > 0.5 : c[0] === "live_change" ? null : v > 0;
      return `<td class="${good == null || v == null ? "" : good ? "good" : "bad"}">${c[3](v)}</td>`;
    }).join("") + "</tr>";
  }).join("");
  $("lb-table").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

$("lb-table").addEventListener("click", (e) => {
  const th = e.target.closest("th");
  if (th) { lb.sort = th.dataset.k; renderLeaderboard(); return; }
  const tr = e.target.closest("tr[data-s]");
  if (tr) {
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (window.COINBAND_STATIC) {
      const row = lb.rows.find((r) => r.symbol === tr.dataset.s);
      $("showcase").value = row.file;
      loadStatic(row.file);
      return;
    }
    $("symbol").value = tr.dataset.s;
    runBacktest();
  }
});

async function runLeaderboard(ev) {
  ev.preventDefault();
  const el = $("lb-error");
  el.hidden = true;
  saveForm();
  const body = formBody();
  body.symbols = $("lb-symbols").value;
  $("lb-run").disabled = true;
  $("lb-progress").textContent = "Submitting…";
  try {
    if (myJobs.leaderboard) cancelJob(myJobs.leaderboard);
    const result = await runJob("/api/leaderboard", body, (job) => ($("lb-progress").textContent = phaseLabel(job)), "leaderboard");
    lb.rows = result.rows;
    lb.trend = result.params.trend_window > 0;
    const p = result.params;
    $("lb-progress").textContent =
      `${result.model.split("/").pop()} · ${result.interval} · ${p.max_origins} origins, ${p.horizon}-bar horizon` +
      (p.covariates ? ", volume covariates" : "") +
      (lb.trend ? `, ${p.trend_window}-bar trend` : "") + " · click a row to replay it";
    renderLeaderboard();
  } catch (e) {
    if (e instanceof Cancelled) {
      $("lb-progress").textContent = "Cancelled";
    } else {
      el.hidden = false;
      el.textContent = e.message;
      $("lb-progress").textContent = "";
    }
  } finally {
    $("lb-run").disabled = false;
  }
}

// ---------- live mode ----------

const liveState = { timer: null, busy: false };

function stopLive() {
  clearInterval(liveState.timer);
  liveState.timer = null;
  $("live").checked = false;
  $("live-status").textContent = "";
}

function startLive() {
  clearInterval(liveState.timer);
  liveTick();
  liveState.timer = setInterval(liveTick, LIVE_EVERY_MS);
}

// x position of the current price: inside the bar that is forming after the last closed one
function livePriceX() {
  const d = state.data;
  const lp = state.livePrice;
  if (!d || !lp || lp.time == null) return null;
  const start = (d.future_time && d.future_time[0]) || d.time[d.n - 1] + d.interval_ms;
  const frac = Math.max(0, Math.min(1, (lp.time - start) / d.interval_ms));
  return d.n - 1 + frac;
}

async function liveTick() {
  const r = state.result;
  if (liveState.busy || !r) return;
  liveState.busy = true;
  $("live-status").textContent = "updating…";
  try {
    const body = { source: r.source, symbol: r.symbol, interval: r.interval, model: r.model, ...r.params, since: r.time[r.time.length - 1] };
    const L = await api("/api/live", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (state.result !== r) return; // a new backtest replaced this one meanwhile
    const nb = L.new_bars;
    const merged = {
      ...r,
      time: r.time.concat(nb.time),
      close: r.close.concat(nb.close),
      high: r.high.concat(nb.high),
      low: r.low.concat(nb.low),
      future_time: L.future_time,
    };
    const live = { ...L.live, index: merged.close.length };
    // new closed bars: the old live forecast becomes a past one; otherwise replace it
    merged.origins = nb.time.length ? r.origins.concat([live]) : r.origins.slice(0, -1).concat([live]);
    state.livePrice = { price: L.price, time: L.price_time };
    loadResult(merged, { live: true });
    const at = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    $("live-status").textContent = `updated ${at}` + (nb.time.length ? ` · ${nb.time.length} new bar${nb.time.length > 1 ? "s" : ""}` : "");
  } catch (e) {
    $("live-status").textContent = `live update failed: ${e.message}`;
  } finally {
    liveState.busy = false;
  }
}

// ---------- wiring ----------

function setView(v) {
  state.view = v;
  $("view-follow").classList.toggle("on", v === "follow");
  $("view-full").classList.toggle("on", v === "full");
}

$("run-form").addEventListener("submit", runBacktest);
$("lb-form").addEventListener("submit", runLeaderboard);
$("source").addEventListener("change", () => applySource(false));
$("cancel-run").addEventListener("click", () => myJobs.backtest && cancelJob(myJobs.backtest));
setInterval(() => document.visibilityState === "visible" && refreshQueue(), 2000);
$("lb-symbols").addEventListener("input", (e) => (e.target.dataset.touched = "1"));
$("live").addEventListener("change", (e) => (e.target.checked ? startLive() : stopLive()));
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

// ---------- static demo (GitHub Pages): precomputed runs, no server ----------

async function loadStatic(file) {
  showError("");
  setProgress("Loading…", 0.3);
  try {
    const r = await (await fetch(file)).json();
    loadResult(r);
    setProgress(`${r.symbol}: ${r.origins.length} hourly forecasts, computed ${r.generated || ""}`, 1);
  } catch (e) {
    showError(`Could not load ${file}: ${e.message}`);
  }
}

async function staticInit() {
  const S = window.COINBAND_STATIC;
  const idx = await (await fetch(S.index)).json();
  const runs = idx.runs;
  $("run-form").innerHTML = `
    <label>Coin <select id="showcase"></select></label>
    <div id="progress" class="progress" hidden><div class="bar"><div id="progress-fill"></div></div><div id="progress-text"></div></div>
    <div id="error" class="error" hidden></div>
    <p class="hint">Precomputed on ${idx.generated} with ${idx.model_name}: a new ${idx.horizon}-hour forecast every hour
      for the last ${idx.days} days, on Binance hourly prices. Scrub, replay and record it here.</p>
    <p class="hint">To run any coin, stock or index, with live forecasts that refresh every minute,
      <a href="${S.repo}">run coinband on your machine</a>.</p>`;
  const sel = $("showcase");
  sel.replaceChildren(...runs.map((r) => Object.assign(document.createElement("option"), { value: r.file, textContent: `${r.symbol}  ${r.summary.crps_skill >= 0 ? "+" : ""}${(r.summary.crps_skill * 100).toFixed(1)}% CRPS skill` })));
  sel.addEventListener("change", () => loadStatic(sel.value));
  $("lb-form").style.display = "none";
  $("live").parentElement.style.display = "none";
  $("live-status").style.display = "none";
  $("queue").hidden = true;
  const link = document.querySelector(".paper-link");
  link.href = S.article;
  link.textContent = "Read the article";
  lb.rows = runs.map((r) => ({ symbol: r.symbol, summary: r.summary, live_change: r.live_change, file: r.file }));
  lb.trend = false;
  renderLeaderboard();
  $("lb-progress").textContent = "";
  document.querySelector(".leaderboard h2 .sub").textContent = `${idx.model_name} · last ${idx.days} days · click a row to replay it`;
  requestAnimationFrame(frame);
  const want = new URLSearchParams(location.hash.slice(1)).get("symbol");
  const first = runs.find((r) => r.symbol === want) || runs[0];
  sel.value = first.file;
  loadStatic(first.file);
}

(async function init() {
  if (window.COINBAND_STATIC) {
    await staticInit();
    return;
  }
  try {
    await loadModels();
  } catch (e) {
    showError(`Server unreachable: ${e.message}`);
  }
  try {
    MARKETS = await api("/api/markets");
  } catch { /* server unreachable, reported above */ }
  applySource(true);
  restoreForm();
  // Shareable links: #symbol=ETHUSDT&interval=4h&horizon=12&run fills the form (and runs it).
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.has("source")) { $("source").value = hash.get("source"); applySource(false); }
  for (const f of FORM_FIELDS) if (hash.has(f) && f !== "source") $(f).value = hash.get(f);
  if (hash.has("covariates")) $("covariates").checked = hash.get("covariates") !== "0";
  if (hash.get("view") === "full") setView("full");
  requestAnimationFrame(frame);
  refreshQueue();
  if (hash.has("run")) $("run-form").requestSubmit();
})();
