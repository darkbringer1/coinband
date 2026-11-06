"""Phase 2: do better inputs make Timestral a better risk model?

Same eight pairs, model (209M, 4096-hour context) and halves as phase 1, but origins every 12 hours (inputs make
inference several times slower). Each configuration adds inputs to the price:

- `market`: the hourly log return and absolute return of the market leader (BTC for the altcoins, ETH for BTC),
  as past-only covariates.
- `funding`: the perpetual-futures funding rate (basis points, forward-filled to hours), past-only. It shows how
  crowded long or short positioning is.
- `calendar`: hour of day and day of week as sine/cosine pairs, known in advance (also over the horizon).
- `volume`: log volume and absolute return of the pair itself, past-only (as in the first study).
- `all`: everything above together.

Scored against plain Timestral at the same origins (from phase 1) and against GARCH (fitted on the tuning half),
on CRPS (whole distribution), QLIKE (24 h volatility) and the 1% one-hour Value-at-Risk.

    uv run python experiments/phase2/inputs.py collect --device mps
    uv run python experiments/phase2/inputs.py evaluate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "phase1"))
from risk import baseline_quantiles, block_resamples, crps, qlike, sigma_from_quantiles  # noqa: E402
from sweep import CACHE, CONFIRM_PAIRS, load_klines  # noqa: E402

from coinband.backtest import BacktestParams, run_backtest  # noqa: E402

RESULTS = HERE.parent / "results" / "phase2"
PHASE1 = HERE.parent / "results" / "phase1"
BARS, CONTEXT, H, STRIDE = 13152, 4096, 24, 12
MODEL = "mistralai/Timestral-209M"
CONFIGS = ("market", "funding", "calendar", "volume", "all")
HOUR_MS = 3_600_000


def funding_history(symbol: str, start_ms: int) -> list[tuple[int, float]]:
    """(fundingTime ms, rate) from Binance USD-M futures, cached locally."""
    path = CACHE / f"funding-{symbol}.json"
    if path.exists():
        return [tuple(x) for x in json.loads(path.read_text())]
    rows, t = [], start_ms
    while True:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={t}&limit=1000"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "coinband"}), timeout=20) as r:
            page = json.load(r)
        rows += [(int(p["fundingTime"]), float(p["fundingRate"])) for p in page]
        if len(page) < 1000:
            break
        t = int(page[-1]["fundingTime"]) + 1
    path.write_text(json.dumps(rows))
    return rows


def inputs_for(symbol: str, data: dict) -> dict:
    """Past (C, T) and known (C, T + H) covariate rows for every configuration, aligned on the pair's bars."""
    times, close, volume = data[symbol]
    n = len(close)
    logc = np.log(close)
    # market leader returns, aligned by timestamp (caches were fetched at slightly different times)
    leader = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
    lt, lc, _ = data[leader]
    lr = dict(zip(lt.tolist(), np.diff(np.log(lc), prepend=np.nan).tolist()))
    mret = np.array([lr.get(int(t), np.nan) for t in times], dtype=np.float64)
    market = np.vstack([mret, np.abs(mret)])
    # funding rate known at the close of each bar, in basis points
    fund = funding_history(symbol, int(times[0]) - 8 * HOUR_MS)
    ft = np.array([f[0] for f in fund])
    fr = np.array([f[1] for f in fund]) * 1e4
    pos = np.searchsorted(ft, times + HOUR_MS - 1, side="right") - 1
    funding = np.where(pos >= 0, fr[np.clip(pos, 0, None)], np.nan)[None, :]
    # calendar over the bars and the horizon after the data end
    all_times = np.concatenate([times, times[-1] + HOUR_MS * np.arange(1, H + 1)])
    dts = [datetime.fromtimestamp(t / 1000, UTC) for t in all_times]
    hour = np.array([d.hour for d in dts]) * 2 * math.pi / 24
    dow = np.array([d.weekday() for d in dts]) * 2 * math.pi / 7
    calendar = np.vstack([np.sin(hour), np.cos(hour), np.sin(dow), np.cos(dow)])
    vol = np.vstack([np.log1p(np.maximum(volume, 0)), np.abs(np.diff(logc, prepend=np.nan))])
    return {
        "market": {"past": market},
        "funding": {"past": funding},
        "calendar": {"known": calendar},
        "volume": {"past": vol},
        "all": {"past": np.vstack([market, funding, vol]), "known": calendar},
    }


def collect(device: str, only: list[str]) -> None:
    from coinband.device import load_forecaster

    model = load_forecaster(MODEL, device)
    data = {p: load_klines(p, "1h", BARS) for p in CONFIRM_PAIRS}
    for cfg in only:
        out = RESULTS / cfg
        out.mkdir(parents=True, exist_ok=True)
        for symbol in CONFIRM_PAIRS:
            path = out / f"{symbol}.npz"
            if path.exists():
                continue
            times, close, _ = data[symbol]
            inp = inputs_for(symbol, data)[cfg]
            n_origins = (len(close) - CONTEXT) // STRIDE + 1
            params = BacktestParams(context=CONTEXT, horizon=H, stride=STRIDE, max_origins=n_origins, batch_size=32)
            started = time.time()
            bt = run_backtest(model, close, params, raw=True, past_covariates=inp.get("past"), known_covariates=inp.get("known"))
            raw = bt["raw"]
            last = close[raw["index"] - 1][:, None, None]
            q = np.log(np.maximum(raw["q"], 1e-12) / last).astype(np.float32)
            tmp = out / f".{symbol}.tmp.npz"
            np.savez_compressed(tmp, index=raw["index"], q=q, levels=raw["levels"])
            tmp.replace(path)
            print(f"{cfg} {symbol}: {len(raw['index'])} origins in {time.time() - started:.0f}s", flush=True)


def evaluate() -> None:
    rng = np.random.default_rng(0)
    data = {p: load_klines(p, "1h", BARS) for p in CONFIRM_PAIRS}
    configs = [c for c in CONFIGS if all((RESULTS / c / f"{p}.npz").exists() for p in CONFIRM_PAIRS)]
    rows = {}
    per = {c: {} for c in configs}
    for p in CONFIRM_PAIRS:
        times, close, _ = data[p]
        logc = np.log(close)
        p1 = np.load(PHASE1 / f"{p}.npz")
        idx1 = p1["index"][p1["index"] + H <= len(close)]
        fit_end = int(idx1[len(idx1) // 2])
        levels = p1["levels"]
        for c in configs:
            f = np.load(RESULTS / c / f"{p}.npz")
            keep = f["index"] + H <= len(close)
            idx, qc = f["index"][keep], f["q"][keep].astype(np.float64)
            pos = np.searchsorted(p1["index"], idx)
            qp = p1["q"][pos].astype(np.float64)
            qg = baseline_quantiles(logc, idx, levels, fit_end)["garch"]
            y = np.stack([logc[i : i + H] - logc[i - 1] for i in idx])
            rv = np.cumsum(np.stack([np.diff(logc)[i - 1 : i - 1 + H] ** 2 for i in idx]), axis=1)
            li = int(np.argmin(np.abs(levels - 0.01)))
            per[c][p] = {
                "half": idx >= fit_end,
                "crps": {m: crps(q, y, levels) for m, q in (("cfg", qc), ("plain", qp), ("garch", qg))},
                "qlike": {m: (rv[:, -1], sigma_from_quantiles(q, levels)[:, -1] ** 2) for m, q in (("cfg", qc), ("plain", qp), ("garch", qg))},
                "var": {m: (y[:, 0], q[:, 0, li]) for m, q in (("cfg", qc), ("plain", qp), ("garch", qg))},
            }
    lines = ["CRPS skill of each input configuration vs plain Timestral and vs GARCH, % (90% CI); QLIKE of the 24 h",
             "volatility and the 1% one-hour VaR hit rate (target 1%). Origins every 12 hours, eight pairs.", ""]
    for half_name, flag in (("tuning", False), ("heldout", True)):
        lines += [f"## {half_name}", "", "| inputs | vs plain h=1 | vs plain h=24 | vs plain avg (90% CI) | vs GARCH avg (90% CI) | QLIKE inputs / plain / GARCH | VaR 1% hits |",
                  "|---|---|---|---|---|---|---|"]
        rows[half_name] = {}
        for c in configs:
            def pooled(model, ref, steps=slice(None)):
                num = sum(per[c][p]["crps"][model][per[c][p]["half"] == flag][:, steps].sum() for p in CONFIRM_PAIRS)
                den = sum(per[c][p]["crps"][ref][per[c][p]["half"] == flag][:, steps].sum() for p in CONFIRM_PAIRS)
                return 1 - num / den

            def boot(model, ref):
                ns, ds = np.zeros(1000), np.zeros(1000)
                for p in CONFIRM_PAIRS:
                    sel = per[c][p]["half"] == flag
                    a = per[c][p]["crps"][model][sel].mean(axis=1)
                    b = per[c][p]["crps"][ref][sel].mean(axis=1)
                    ix = block_resamples(len(a), 4, rng)
                    ns += a[ix].mean(axis=1)
                    ds += b[ix].mean(axis=1)
                d = 1 - ns / ds
                return np.percentile(d, 5), np.percentile(d, 95)

            vp, vg = pooled("cfg", "plain"), pooled("cfg", "garch")
            (pl, ph), (gl, gh) = boot("cfg", "plain"), boot("cfg", "garch")
            ql = {}
            for m in ("cfg", "plain", "garch"):
                ql[m] = np.mean([qlike(per[c][p]["qlike"][m][0][per[c][p]["half"] == flag], per[c][p]["qlike"][m][1][per[c][p]["half"] == flag]) for p in CONFIRM_PAIRS])
            hits = np.mean(np.concatenate([(lambda yv, qv, s: (yv[s] < qv[s]))(*per[c][p]["var"]["cfg"], per[c][p]["half"] == flag) for p in CONFIRM_PAIRS]))
            rows[half_name][c] = {"vs_plain": vp, "vs_plain_ci": [pl, ph], "vs_garch": vg, "vs_garch_ci": [gl, gh],
                                  "h1": pooled("cfg", "plain", slice(0, 1)), "h24": pooled("cfg", "plain", slice(23, 24)), "qlike": ql, "var_hits": float(hits)}
            lines.append(f"| {c} | {rows[half_name][c]['h1'] * 100:+.2f} | {rows[half_name][c]['h24'] * 100:+.2f} | {vp * 100:+.2f} ({pl * 100:+.2f}, {ph * 100:+.2f}) | "
                         f"{vg * 100:+.2f} ({gl * 100:+.2f}, {gh * 100:+.2f}) | {ql['cfg']:.3f} / {ql['plain']:.3f} / {ql['garch']:.3f} | {hits * 100:.2f}% |")
        plain_vs_g = 1 - sum(per[configs[0]][p]["crps"]["plain"][per[configs[0]][p]["half"] == flag].sum() for p in CONFIRM_PAIRS) / \
            sum(per[configs[0]][p]["crps"]["garch"][per[configs[0]][p]["half"] == flag].sum() for p in CONFIRM_PAIRS)
        lines += ["", f"Plain Timestral vs GARCH at these origins: {plain_vs_g * 100:+.2f}%", ""]
        rows[half_name]["plain_vs_garch"] = plain_vs_g
    text = "\n".join(lines)
    (RESULTS / "summary.md").write_text(text + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["collect", "evaluate"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    if args.action == "collect":
        collect(args.device, args.configs.split(","))
    else:
        evaluate()


if __name__ == "__main__":
    main()
