"""Phase 3: Timestral against the options market.

Deribit's DVOL index is the 30-day implied volatility of BTC and ETH options: the options market's own forecast of
how much the price will move. We take it hourly and compare it with the volatility forecasts from phase 1
(Timestral, GARCH, the blend) at forecasting the realized volatility of the next 24 hours.

Caveats built into the comparison: DVOL is a 30-day, risk-neutral number (it carries a premium for insurance), while
the others forecast the next 24 hours. So we look at three things:

1. Forecast loss (QLIKE, lower is better) of each forecast as is, and after a linear recalibration fitted on the
   tuning half (log RV = a + b log forecast), which removes the premium and the horizon mismatch.
2. Encompassing: regress log realized variance on log DVOL and log Timestral variance together (tuning half fit,
   held-out half R^2 and coefficients with a block-bootstrap interval). If Timestral's coefficient stays positive
   next to DVOL, it knows something the options market does not price.
3. The volatility risk premium (implied minus realized variance, what option sellers earn): does "DVOL minus the
   model's forecast" predict it? Correlation on the held-out half.
4. A variance-selling proxy: each day, sell one day of variance at the implied level (DVOL^2 / 365) and pay the
   realized variance, like a one-day variance swap. Compare always selling with selling only on days when DVOL is
   far above a model's forecast (threshold: the tuning-half median of the gap, or, as a robustness check, the gap's
   own trailing 90-day median). This ignores the 30-day term
   structure, bid-ask spreads and hedging costs, so it shows timing value, not attainable profits.

    uv run python experiments/phase3/options.py
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "phase1"))
from risk import baseline_quantiles, qlike, sigma_from_quantiles  # noqa: E402
from sweep import CACHE, load_klines  # noqa: E402

PHASE1 = HERE.parent / "results" / "phase1"
OUT = HERE.parent / "results" / "phase3"
BARS, H = 13152, 24
PAIRS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
N_BOOT = 1000


def dvol_history(currency: str, start_ms: int, end_ms: int) -> dict[int, float]:
    """Hourly DVOL closes {open time ms: annualized vol in %}, cached locally."""
    path = CACHE / f"dvol-{currency}.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    out: dict[int, float] = {}
    end = end_ms
    while end > start_ms:
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
               f"?currency={currency}&start_timestamp={start_ms}&end_timestamp={end}&resolution=3600")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "coinband"}), timeout=30) as r:
            res = json.load(r)["result"]
        rows = res["data"]
        if not rows:
            break
        for t, _o, _h, _l, c in rows:
            out[int(t)] = float(c)
        nxt = res.get("continuation")
        if not nxt or nxt >= end:
            break
        end = int(nxt)
    path.write_text(json.dumps(out))
    return out


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = ["Realized variance of the next 24 hours (sum of squared hourly log returns) against forecasts from",
             "Deribit DVOL (30-day implied vol), GARCH, Timestral and the blend. Baselines and recalibrations fitted on",
             "the tuning half; scores on the held-out half unless noted. Daily, non-overlapping origins.", ""]
    summary = {}
    for symbol, cur in PAIRS.items():
        times, close, _ = load_klines(symbol, "1h", BARS)
        logc = np.log(close)
        f = np.load(PHASE1 / f"{symbol}.npz")
        idx_all, levels = f["index"], f["levels"]
        keep = idx_all + H <= len(close)
        idx_all, q_all = idx_all[keep], f["q"][keep].astype(np.float64)
        fit_end = int(idx_all[len(idx_all) // 2])
        dv = dvol_history(cur, int(times[0]), int(times[-1]) + 3_600_000)
        # one origin per day (non-overlapping 24 h windows), where DVOL is known at the origin
        sel = np.arange(0, len(idx_all), 24)
        idx, q_t = idx_all[sel], q_all[sel]
        # DVOL at the close of the bar before the origin: the hourly candle that opened one hour earlier
        dvol = np.array([dv.get(int(times[i - 1]), np.nan) for i in idx])
        ok = np.isfinite(dvol)
        idx, q_t, dvol = idx[ok], q_t[ok], dvol[ok]
        base = baseline_quantiles(logc, idx, levels, fit_end)
        q_b = 0.5 * (q_t + base["garch"])
        rv = np.array([np.sum(np.diff(logc)[i - 1 : i - 1 + H] ** 2) for i in idx])
        fc = {
            "DVOL (implied)": (dvol / 100) ** 2 / 365,  # 30-day annualized vol -> daily variance, flat term structure
            "GARCH": sigma_from_quantiles(base["garch"], levels)[:, -1] ** 2,
            "Timestral": sigma_from_quantiles(q_t, levels)[:, -1] ** 2,
            "Blend": sigma_from_quantiles(q_b, levels)[:, -1] ** 2,
        }
        tune = idx < fit_end
        held = ~tune
        y = np.log(np.maximum(rv, 1e-12))
        res: dict = {"n_days_tuning": int(tune.sum()), "n_days_heldout": int(held.sum())}
        lines += [f"## {symbol} (DVOL {cur}), {int(tune.sum())} tuning days, {int(held.sum())} held-out days", "",
                  "| forecast | QLIKE raw | QLIKE recalibrated | Mincer-Zarnowitz R² (log) | slope b |", "|---|---|---|---|---|"]
        res["forecasts"] = {}
        for name, v in fc.items():
            lv = np.log(v)
            b = ols(lv[tune], y[tune])
            pred = b[0] + b[1] * lv
            recal = np.exp(pred + 0.5 * np.var(y[tune] - (b[0] + b[1] * lv[tune])))  # back to variance, lognormal correction
            res["forecasts"][name] = {"qlike_raw": qlike(rv[held], v[held]), "qlike_recal": qlike(rv[held], recal[held]),
                                      "r2_heldout": r2(y[held], pred[held]), "slope": float(b[1])}
            r = res["forecasts"][name]
            lines.append(f"| {name} | {r['qlike_raw']:.3f} | {r['qlike_recal']:.3f} | {r['r2_heldout']:.3f} | {r['slope']:.2f} |")
        # encompassing: log RV on log DVOL and log Timestral together
        X = np.column_stack([np.log(fc["DVOL (implied)"]), np.log(fc["Timestral"])])
        b = ols(X[tune], y[tune])
        pred = b[0] + X @ b[1:]
        r2_both = r2(y[held], pred[held])
        nb = int(held.sum())
        hx, hy = X[held], y[held]
        boots = []
        for _ in range(N_BOOT):
            start = rng.integers(0, max(1, nb - 7), size=math.ceil(nb / 7))
            ix = (start[:, None] + np.arange(7)).ravel()[:nb]
            boots.append(ols(hx[ix], hy[ix])[1:])
        boots = np.array(boots)
        bh = ols(hx, hy)
        res["encompassing"] = {"tuning_fit": b.tolist(), "r2_heldout_both": r2_both,
                               "heldout_coef": bh[1:].tolist(),
                               "heldout_ci": [np.percentile(boots, 5, axis=0).tolist(), np.percentile(boots, 95, axis=0).tolist()]}
        lines += ["", f"Encompassing, log RV = a + b1·log DVOL + b2·log Timestral: held-out R² {r2_both:.3f} "
                  f"(DVOL alone {res['forecasts']['DVOL (implied)']['r2_heldout']:.3f}, Timestral alone {res['forecasts']['Timestral']['r2_heldout']:.3f}); "
                  f"held-out coefficients b1 = {bh[1]:+.2f} ({np.percentile(boots[:, 0], 5):+.2f}, {np.percentile(boots[:, 0], 95):+.2f}), "
                  f"b2 = {bh[2]:+.2f} ({np.percentile(boots[:, 1], 5):+.2f}, {np.percentile(boots[:, 1], 95):+.2f})", ""]
        # volatility risk premium: implied minus realized (log), predicted by implied minus model forecast
        vrp = np.log(fc["DVOL (implied)"]) - y
        lines += ["| VRP signal: log DVOL − log model forecast | correlation with realized VRP, held-out (90% CI) |", "|---|---|"]
        res["vrp"] = {}
        for name in ("GARCH", "Timestral", "Blend"):
            sig = np.log(fc["DVOL (implied)"]) - np.log(fc[name])
            c = float(np.corrcoef(sig[held], vrp[held])[0, 1])
            cb = []
            for _ in range(N_BOOT):
                start = rng.integers(0, max(1, nb - 7), size=math.ceil(nb / 7))
                ix = (start[:, None] + np.arange(7)).ravel()[:nb]
                cb.append(np.corrcoef(sig[held][ix], vrp[held][ix])[0, 1])
            res["vrp"][name] = {"corr": c, "ci": [float(np.percentile(cb, 5)), float(np.percentile(cb, 95))]}
            lines.append(f"| {name} | {c:+.3f} ({np.percentile(cb, 5):+.3f}, {np.percentile(cb, 95):+.3f}) |")
        lines.append("")
        lines.append(f"Average implied minus realized variance (log), held-out: {vrp[held].mean():+.2f} "
                     f"(implied daily vol / realized daily vol = {math.exp(vrp[held].mean() / 2):.2f}x)")
        # 4. variance-selling proxy, P&L in daily variance points scaled to annualized vol-points^2 / 1e4
        pnl_unit = fc["DVOL (implied)"] - rv  # sell 1 unit of one-day variance
        scale = 365 * 1e4  # express in annualized variance points (vol% squared) per day
        lines += ["", "Variance-selling proxy, held-out: P&L per day in annualized variance points (vol% squared),",
                  "selling one day of variance at DVOL and paying realized; Sharpe annualized from daily P&L.", "",
                  "| rule | days short | mean P&L/day | Sharpe (90% CI) | worst day | P&L per day sold |", "|---|---|---|---|---|---|"]
        res["vrp_strategy"] = {}
        rules = {"always sell": np.ones(len(rv), dtype=bool)}
        for name in ("GARCH", "Timestral", "Blend"):
            gap = np.log(fc["DVOL (implied)"]) - np.log(fc[name])
            rules[f"sell when DVOL ≫ {name}"] = gap >= np.median(gap[tune])
        for name in ("GARCH", "Timestral", "Blend"):
            # robustness: compare each day's gap with its own trailing 90-day median, so a level shift cannot
            # switch the rule off (uses only past days)
            gap = np.log(fc["DVOL (implied)"]) - np.log(fc[name])
            trailing = np.array([np.median(gap[max(0, t - 90) : t]) if t >= 30 else np.inf for t in range(len(gap))])
            rules[f"sell when gap > its 90-day median, {name}"] = gap > trailing
        for rule, on in rules.items():
            pnl = np.where(on, pnl_unit, 0.0)[held] * scale
            sh = pnl.mean() / pnl.std() * math.sqrt(365) if pnl.std() > 0 else 0.0
            shb = []
            for _ in range(N_BOOT):
                start = rng.integers(0, max(1, nb - 7), size=math.ceil(nb / 7))
                ix = (start[:, None] + np.arange(7)).ravel()[:nb]
                shb.append(pnl[ix].mean() / pnl[ix].std() * math.sqrt(365))
            days = int(on[held].sum())
            res["vrp_strategy"][rule] = {"days": days, "mean": float(pnl.mean()), "sharpe": float(sh),
                                         "ci": [float(np.percentile(shb, 5)), float(np.percentile(shb, 95))],
                                         "worst": float(pnl.min()), "per_day_sold": float(pnl.sum() / max(days, 1))}
            r = res["vrp_strategy"][rule]
            lines.append(f"| {rule} | {days} | {r['mean']:+.1f} | {sh:+.2f} ({r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}) | {r['worst']:+.0f} | {r['per_day_sold']:+.1f} |")
        lines.append("")
        summary[symbol] = res
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    main()
