"""Phase 4, risk: is Timestral a better risk model than GARCH on stocks, indices and FX? And how does it compare with
the VIX?

The part 2 study (`phase1/risk.py`), repeated on Yahoo data. A forecast at every bar; each model forecasts the
next H log returns as 21 quantiles. Origins are split in time: the first half tunes the baselines (random walk,
historical simulation, EWMA, GARCH(1,1), the same code as part 2), the second half is held out. Timestral is used as
released; its settings are the crypto defaults or the nearest that fit Yahoo's history.

Groups (USD/TRY is left out: its managed, steady devaluation makes any driftless baseline look bad):

- `fx-1h`: EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF and gold futures, hourly. 4096-bar context, 24 hours ahead.
- `stocks-1h`: US indices and large caps plus BIST, hourly (only about two years exist). 1024-bar context, 7 bars
  ahead (about one US trading day).
- `daily`: all of the above, daily bars, the last 3000 days (about twelve years). 2048-day context, 21 trading days
  ahead (one month, the VIX horizon).

Tests: CRPS (whole distribution) and QLIKE (volatility) against every baseline, and the left-tail Value-at-Risk
(1% and 5%) at one bar and at the horizon (10 days for `daily`, the Basel horizon), with Kupiec tests.

`vix`: the S&P 500 against the VIX and the Nasdaq-100 against the VXN, the 30-day implied volatility of their
options. Target: realized variance of the next 21 trading days (sum of squared daily log returns). Like part 3:
loss and Mincer-Zarnowitz R² of each forecast (raw and recalibrated on the tuning half), encompassing (does
Timestral add to the VIX?) and whether "implied minus forecast" predicts the variance risk premium.

Bars are pinned in the snapshot under `~/.cache/coinband/experiments` (see `markets.py`), never in the repository.

    uv run python experiments/phase4/markets_risk.py collect --device mps
    uv run python experiments/phase4/markets_risk.py evaluate
    uv run python experiments/phase4/markets_risk.py vix
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "phase1"))
import risk as part2  # noqa: E402
from markets import FX, STOCKS, get_model, load_bars, snapshot_path  # noqa: E402

from coinband.backtest import BacktestParams, run_backtest  # noqa: E402

OUT = HERE.parent / "results" / "phase4" / "risk"
N_BOOT = 1000
FX_RISK = [s for s in FX if s != "TRY=X"]
BASELINES = ("rw", "hs", "ewma", "garch")
MODELS = ("timestral",) + BASELINES + ("blend",)

GROUPS: dict[str, dict] = {
    "fx-1h": {"symbols": FX_RISK, "interval": "1h", "context": 4096, "H": 24, "steps": (1, 8, 24), "var_k": 24,
              "max_origins": None, "unit": "hour"},
    "stocks-1h": {"symbols": STOCKS, "interval": "1h", "context": 1024, "H": 7, "steps": (1, 3, 7), "var_k": 7,
                  "max_origins": None, "unit": "bar"},
    "daily": {"symbols": STOCKS + FX_RISK, "interval": "1d", "context": 2048, "H": 21, "steps": (1, 5, 21),
              "var_k": 10, "max_origins": 3000, "unit": "day"},
}
VIX_PAIRS = {"^GSPC": "^VIX", "^NDX": "^VXN"}


def series(symbol: str, interval: str, group: dict) -> tuple[np.ndarray, np.ndarray]:
    """Times and closes, trimmed to what the group's origins need (keeps the GARCH fit on the same years)."""
    close = load_bars(symbol, interval)
    times = np.asarray(json.loads(snapshot_path(symbol, interval).read_text())["time"])
    if group["max_origins"]:
        keep = group["context"] + group["max_origins"] + group["H"]
        times, close = times[-keep:], close[-keep:]
    return times, close


def collect(device: str, only: list[str] | None) -> None:
    model = get_model("209M", device)
    for name, g in GROUPS.items():
        if only and name not in only:
            continue
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        for symbol in g["symbols"]:
            path = out / f"{symbol}.npz"
            if path.exists():
                print(f"skip {name}/{symbol}", flush=True)
                continue
            _, close = series(symbol, g["interval"], g)
            n_origins = len(close) - g["context"] + 1
            params = BacktestParams(context=g["context"], horizon=g["H"], stride=1, max_origins=n_origins, batch_size=64)
            started = time.time()
            raw = run_backtest(model, close, params, raw=True)["raw"]
            last = close[raw["index"] - 1][:, None, None]
            q = np.log(np.maximum(raw["q"], 1e-12) / last).astype(np.float32)
            tmp = out / f".{symbol}.tmp.npz"
            np.savez_compressed(tmp, index=raw["index"], q=q, levels=raw["levels"])
            tmp.replace(path)
            print(f"{name}/{symbol}: {len(raw['index'])} origins in {time.time() - started:.0f}s", flush=True)


def load_forecasts(name: str, symbol: str) -> dict:
    """Timestral and every baseline as k-bar log-return quantiles at the realized origins, split in halves."""
    g = GROUPS[name]
    part2.H = g["H"]  # the part 2 baselines read the horizon from their module
    times, close = series(symbol, g["interval"], g)
    logc = np.log(close)
    f = np.load(OUT / name / f"{symbol}.npz")
    idx, q_t, levels = f["index"], f["q"].astype(np.float64), f["levels"]
    keep = idx + g["H"] <= len(close)
    idx, q_t = idx[keep], q_t[keep]
    half = len(idx) // 2
    fit_end = int(idx[half])
    base = part2.baseline_quantiles(logc, idx, levels, fit_end)
    params = base.pop("_params")
    quant = {"timestral": q_t, **base, "blend": 0.5 * (q_t + base["garch"])}
    y = np.stack([logc[i : i + g["H"]] - logc[i - 1] for i in idx])
    r = np.diff(logc)
    rv = np.cumsum(np.stack([r[i - 1 : i - 1 + g["H"]] ** 2 for i in idx]), axis=1)
    return {"idx": idx, "times": times, "quant": quant, "levels": levels, "y": y, "rv": rv, "half": half,
            "params": params}


def evaluate_group(name: str) -> tuple[dict, list[str]]:
    g = GROUPS[name]
    H, steps, kv = g["H"], g["steps"], g["var_k"]
    rng = np.random.default_rng(0)
    data = {s: load_forecasts(name, s) for s in g["symbols"]}
    lines = [f"## {name}: {', '.join(g['symbols'])}", ""]
    res: dict = {"params": {s: d["params"] for s, d in data.items()}, "periods": {}}
    for period in ("tuning", "heldout"):
        sl = {s: (slice(0, d["half"]) if period == "tuning" else slice(d["half"], None)) for s, d in data.items()}
        P: dict = {"dates": {}}
        crps_rows, crps_steps, ql = {}, {}, {}
        for s, d in data.items():
            yy, lv = d["y"][sl[s]], d["levels"]
            crps_rows[s], crps_steps[s], ql[s] = {}, {}, {}
            for m in MODELS:
                c = part2.crps(d["quant"][m][sl[s]], yy, lv)
                crps_rows[s][m] = c.mean(axis=1)
                crps_steps[s][m] = c.sum(axis=0)
                sig = part2.sigma_from_quantiles(d["quant"][m][sl[s]], lv)
                ql[s][m] = part2.qlike(d["rv"][sl[s]][:, H - 1], sig[:, H - 1] ** 2)
            ix = d["idx"][sl[s]]
            P["dates"][s] = [int(d["times"][ix[0]]), int(d["times"][ix[-1]])]
        n_orig = sum(len(next(iter(v.values()))) for v in crps_rows.values())
        lines += [f"### {period} ({n_orig} origins in all)", "",
                  f"CRPS skill vs baseline, % (90% CI on the average over 1..{H}); QLIKE of the {H}-{g['unit']} "
                  "volatility, mean over symbols, lower is better", "",
                  "| model vs baseline | " + " | ".join(f"h={k}" for k in steps) + f" | avg 1-{H} (90% CI) "
                  "| QLIKE baseline | QLIKE model | symbols better |", "|---|" + "---|" * (len(steps) + 4)]
        P["crps_skill"] = {}
        for model, base in [("timestral", b) for b in BASELINES] + [("blend", "garch")]:
            cells = [1 - sum(crps_steps[s][model][k - 1] for s in data) / sum(crps_steps[s][base][k - 1] for s in data)
                     for k in steps]
            nsum, dsum = np.zeros(N_BOOT), np.zeros(N_BOOT)
            for s in data:
                rows_m, rows_b = crps_rows[s][model], crps_rows[s][base]
                ix = part2.block_resamples(len(rows_m), 2 * H, rng)
                nsum += rows_m[ix].mean(axis=1)
                dsum += rows_b[ix].mean(axis=1)
            boots = 1 - nsum / dsum
            avg = 1 - sum(crps_rows[s][model].mean() for s in data) / sum(crps_rows[s][base].mean() for s in data)
            per = {s: float(1 - crps_rows[s][model].mean() / crps_rows[s][base].mean()) for s in data}
            ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
            qb, qm = float(np.mean([ql[s][base] for s in data])), float(np.mean([ql[s][model] for s in data]))
            better = sum(v > 0 for v in per.values())
            P["crps_skill"][f"{model}_vs_{base}"] = {"steps": dict(zip(steps, cells)), "avg": float(avg), "ci": ci,
                                                    "per_symbol": per, "qlike_base": qb, "qlike_model": qm}
            lines.append(f"| {model} vs {base} | " + " | ".join(f"{c * 100:+.2f}" for c in cells)
                         + f" | {avg * 100:+.2f} ({ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}) | {qb:.4f} | {qm:.4f} "
                         f"| {better}/{len(data)} |")
        P["qlike"] = {m: float(np.mean([ql[s][m] for s in data])) for m in MODELS}
        P["qlike_wins_vs_garch"] = {m: int(sum(ql[s][m] < ql[s]["garch"] for s in data)) for m in MODELS}
        # Value-at-Risk, left tail, non-overlapping windows
        lines += ["", f"Value-at-Risk (left tail), hit rates pooled over symbols; Kupiec rejections at 5%; "
                  "quantile loss ×1e4, lower is better", "",
                  f"| model | 1%@1 hits | rej. | qloss | 5%@1 hits | rej. | qloss | 1%@{kv} hits | rej. | 5%@{kv} hits "
                  f"| rej. | qloss 5%@{kv} |", "|---|" + "---|" * 11]
        P["var"] = {}
        for m in MODELS:
            row = {}
            for p in (0.01, 0.05):
                for k in (1, kv):
                    hits_all, rej, qls = [], 0, []
                    for s, d in data.items():
                        li = int(np.argmin(np.abs(d["levels"] - p)))
                        yy, qq = d["y"][sl[s]], d["quant"][m][sl[s]]
                        sel = np.arange(0, len(yy), k)
                        hits = yy[sel, k - 1] < qq[sel, k - 1, li]
                        diff = yy[sel, k - 1] - qq[sel, k - 1, li]
                        hits_all.append(hits)
                        rej += part2.kupiec(hits, p) < 0.05
                        qls.append(float(np.mean(np.maximum(p * diff, (p - 1) * diff))))
                    row[f"{int(p * 100)}%@{k}"] = {"hit_rate": float(np.concatenate(hits_all).mean()),
                                                    "rejections": int(rej), "qloss": float(np.mean(qls))}
            P["var"][m] = row
            a, b, c, e = row["1%@1"], row["5%@1"], row[f"1%@{kv}"], row[f"5%@{kv}"]
            n = len(data)
            lines.append(f"| {m} | {a['hit_rate'] * 100:.2f}% | {a['rejections']}/{n} | {a['qloss'] * 1e4:.3f} "
                         f"| {b['hit_rate'] * 100:.2f}% | {b['rejections']}/{n} | {b['qloss'] * 1e4:.3f} "
                         f"| {c['hit_rate'] * 100:.2f}% | {c['rejections']}/{n} | {e['hit_rate'] * 100:.2f}% "
                         f"| {e['rejections']}/{n} | {e['qloss'] * 1e4:.2f} |")
        # per-step skill vs GARCH, for the figure
        P["curve_vs_garch"] = {m: [float(1 - sum(crps_steps[s][m][k] for s in data)
                                         / sum(crps_steps[s]["garch"][k] for s in data)) for k in range(H)]
                               for m in ("timestral", "blend", "ewma", "rw")}
        lines.append("")
        res["periods"][period] = P
    return res, lines


def evaluate() -> None:
    summary, lines = {}, ["Baselines fitted on the tuning half only. Timestral-209M as released.", ""]
    for name in GROUPS:
        if not all((OUT / name / f"{s}.npz").exists() for s in GROUPS[name]["symbols"]):
            lines.append(f"## {name}: not collected\n")
            continue
        summary[name], part = evaluate_group(name)
        lines += part
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(text)


# ---------------------------------------------------------------- VIX


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def day(ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


def block_ix(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    start = rng.integers(0, max(1, n - block), size=math.ceil(n / block))
    return (start[:, None] + np.arange(block)).ravel()[:n]


def vix() -> None:
    g = GROUPS["daily"]
    H = g["H"]
    rng = np.random.default_rng(0)
    lines = [f"Realized variance of the next {H} trading days (sum of squared daily log returns) against the options",
             "market's 30-day implied volatility (VIX for the S&P 500, VXN for the Nasdaq-100), GARCH, Timestral and the",
             "blend. Fits and recalibrations on the tuning half, scores on the held-out half. Origins every day, so",
             f"targets overlap; intervals use {H}-day blocks.", ""]
    summary = {}
    for symbol, vol_symbol in VIX_PAIRS.items():
        d = load_forecasts("daily", symbol)
        load_bars(vol_symbol, "1d")
        snap = json.loads(snapshot_path(vol_symbol, "1d").read_text())
        implied = {day(t): c for t, c in zip(snap["time"], snap["close"])}
        # implied vol at the close of the bar before each origin (same trading day)
        iv = np.array([implied.get(day(int(d["times"][i - 1])), np.nan) for i in d["idx"]])
        ok = np.isfinite(iv)
        half_mask = np.arange(len(d["idx"])) < d["half"]
        lv = d["levels"]
        rv = d["rv"][ok][:, H - 1]
        fc = {
            "implied": (iv[ok] / 100) ** 2 * H / 252,
            "GARCH": part2.sigma_from_quantiles(d["quant"]["garch"][ok], lv)[:, H - 1] ** 2,
            "Timestral": part2.sigma_from_quantiles(d["quant"]["timestral"][ok], lv)[:, H - 1] ** 2,
            "Blend": part2.sigma_from_quantiles(d["quant"]["blend"][ok], lv)[:, H - 1] ** 2,
        }
        tune, held = half_mask[ok], ~half_mask[ok]
        y = np.log(np.maximum(rv, 1e-12))
        dates = [day(int(d["times"][i])) for i in d["idx"][ok]]
        res: dict = {"vol_index": vol_symbol, "n_tuning": int(tune.sum()), "n_heldout": int(held.sum()),
                     "tuning_dates": [dates[0], dates[int(tune.sum()) - 1]],
                     "heldout_dates": [dates[int(tune.sum())], dates[-1]], "forecasts": {}}
        lines += [f"## {symbol} vs {vol_symbol}: tuning {res['tuning_dates'][0]} to {res['tuning_dates'][1]}, "
                  f"held-out {res['heldout_dates'][0]} to {res['heldout_dates'][1]} ({int(held.sum())} days)", "",
                  "| forecast | QLIKE raw | QLIKE recalibrated | R² (log RV), held-out (90% CI) | slope b |",
                  "|---|---|---|---|---|"]
        nb = int(held.sum())
        boot_ix = [block_ix(nb, H, rng) for _ in range(N_BOOT)]
        for name, v in fc.items():
            lvv = np.log(v)
            b = ols(lvv[tune], y[tune])
            pred = b[0] + b[1] * lvv
            recal = np.exp(pred + 0.5 * np.var(y[tune] - pred[tune]))
            yh, ph = y[held], pred[held]
            r2b = [r2(yh[ix], ph[ix]) for ix in boot_ix]
            res["forecasts"][name] = {"qlike_raw": part2.qlike(rv[held], v[held]),
                                      "qlike_recal": part2.qlike(rv[held], recal[held]),
                                      "r2_heldout": r2(yh, ph), "r2_ci": [float(np.percentile(r2b, 5)),
                                                                           float(np.percentile(r2b, 95))],
                                      "slope": float(b[1])}
            r = res["forecasts"][name]
            lines.append(f"| {name} | {r['qlike_raw']:.3f} | {r['qlike_recal']:.3f} | {r['r2_heldout']:.3f} "
                         f"({r['r2_ci'][0]:.3f}, {r['r2_ci'][1]:.3f}) | {r['slope']:.2f} |")
        # paired R² difference, Timestral and blend vs implied
        res["r2_diff_vs_implied"] = {}
        for name in ("Timestral", "Blend", "GARCH"):
            pi = {}
            for nm in (name, "implied"):
                lvv = np.log(fc[nm])
                b = ols(lvv[tune], y[tune])
                pi[nm] = (b[0] + b[1] * lvv)[held]
            diffs = [r2(y[held][ix], pi[name][ix]) - r2(y[held][ix], pi["implied"][ix]) for ix in boot_ix]
            point = r2(y[held], pi[name]) - r2(y[held], pi["implied"])
            res["r2_diff_vs_implied"][name] = {"point": point, "ci": [float(np.percentile(diffs, 5)),
                                                                      float(np.percentile(diffs, 95))]}
            c = res["r2_diff_vs_implied"][name]
            lines.append(f"\nR² difference, {name} minus implied: {point:+.3f} ({c['ci'][0]:+.3f}, {c['ci'][1]:+.3f})")
        # encompassing
        X = np.column_stack([np.log(fc["implied"]), np.log(fc["Timestral"])])
        bt = ols(X[tune], y[tune])
        r2_both = r2(y[held], (bt[0] + X @ bt[1:])[held])
        bh = ols(X[held], y[held])
        boots = np.array([ols(X[held][ix], y[held][ix])[1:] for ix in boot_ix])
        res["encompassing"] = {"r2_heldout_both": r2_both, "heldout_coef": bh[1:].tolist(),
                               "heldout_ci": [np.percentile(boots, 5, axis=0).tolist(),
                                              np.percentile(boots, 95, axis=0).tolist()]}
        lines += ["", f"Encompassing, log RV = a + b1·log implied + b2·log Timestral: held-out R² {r2_both:.3f}; "
                  f"held-out b1 = {bh[1]:+.2f} ({np.percentile(boots[:, 0], 5):+.2f}, {np.percentile(boots[:, 0], 95):+.2f}), "
                  f"b2 = {bh[2]:+.2f} ({np.percentile(boots[:, 1], 5):+.2f}, {np.percentile(boots[:, 1], 95):+.2f})", ""]
        # variance risk premium timing
        vrp = np.log(fc["implied"]) - y
        res["vrp_mean_heldout"] = float(vrp[held].mean())
        lines += ["| VRP signal: log implied − log model forecast | correlation with realized VRP, held-out (90% CI) |",
                  "|---|---|"]
        res["vrp"] = {}
        for name in ("GARCH", "Timestral", "Blend"):
            sig = np.log(fc["implied"]) - np.log(fc[name])
            c = float(np.corrcoef(sig[held], vrp[held])[0, 1])
            cb = [np.corrcoef(sig[held][ix], vrp[held][ix])[0, 1] for ix in boot_ix]
            res["vrp"][name] = {"corr": c, "ci": [float(np.percentile(cb, 5)), float(np.percentile(cb, 95))]}
            lines.append(f"| {name} | {c:+.3f} ({res['vrp'][name]['ci'][0]:+.3f}, {res['vrp'][name]['ci'][1]:+.3f}) |")
        lines += ["", f"Average log(implied / realized variance), held-out: {vrp[held].mean():+.2f} "
                  f"(implied vol / realized vol = {math.exp(vrp[held].mean() / 2):.2f}x)", ""]
        summary[symbol] = res
    text = "\n".join(lines)
    (OUT / "vix.md").write_text(text + "\n")
    (OUT / "vix.json").write_text(json.dumps(summary, indent=1, default=float))
    print(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--device", default="cpu")
    c.add_argument("--groups", nargs="*", choices=list(GROUPS))
    sub.add_parser("evaluate")
    sub.add_parser("vix")
    args = ap.parse_args(argv)
    {"collect": lambda: collect(args.device, args.groups), "evaluate": evaluate, "vix": vix}[args.cmd]()


if __name__ == "__main__":
    main()
