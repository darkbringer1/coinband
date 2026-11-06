"""Phase 1: is Timestral a better risk model than the standard statistical ones?

Every hour, each model forecasts the next 24 hourly log returns of eight Binance pairs as 21 quantiles. The
origins are split in time: the first half tunes the baselines, the second half is held out.

Baselines (all fitted on the tuning half only):

- `rw`: random walk, normal, sigma from the last 500 hourly returns (the yardstick of the first study).
- `hs`: historical simulation, empirical quantiles of the last 2000 overlapping k-hour returns.
- `ewma`: RiskMetrics-style EWMA variance, lambda fitted by QLIKE; quantiles from the empirical distribution of
  its standardized residuals (filtered historical simulation).
- `garch`: GARCH(1,1) with variance targeting, alpha and beta fitted by Gaussian likelihood; same filtered
  historical simulation for the quantiles, multi-step variance by the usual mean-reverting recursion.
- `blend`: the average of Timestral's and GARCH's quantiles at each level (a combination, not a baseline).

Tests:

1. Forecast quality: CRPS (whole distribution) and QLIKE (volatility) at 1, 8 and 24 hours ahead.
2. Value-at-Risk: hit rates of the 1% and 5% left tails at 1 and 24 hours, Kupiec and Christoffersen tests,
   and the quantile loss at those levels.
3. Volatility targeting: hold each pair (and an equal-weight basket of all eight), resized once a day so the
   forecast 24-hour volatility matches a target, after 10 bp fees on every change in position.

    uv run python experiments/phase1/risk.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sweep import CONFIRM_PAIRS, load_klines  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "phase1"
BARS, H = 13152, 24
HOURS_PER_YEAR = 24 * 365
FEE = 0.001  # 10 bp per unit of position change (Binance taker fee)
MAX_LEVERAGE = 2.0
STEPS = (1, 8, 24)
MODELS = ("timestral", "rw", "hs", "ewma", "garch", "blend")
N_BOOT = 1000


# ---------------------------------------------------------------- baselines


def ewma_var(r: np.ndarray, lam: float, init: float) -> np.ndarray:
    """One-step-ahead variance: out[t] is the forecast for r[t] made after r[t-1]."""
    out = np.empty(len(r))
    v = init
    for t in range(len(r)):
        out[t] = v
        v = lam * v + (1 - lam) * r[t] * r[t]
    return out


def garch_var(r: np.ndarray, alpha: float, beta: float, vbar: float) -> np.ndarray:
    out = np.empty(len(r))
    omega = vbar * (1 - alpha - beta)
    v = vbar
    for t in range(len(r)):
        out[t] = v
        v = omega + alpha * r[t] * r[t] + beta * v
    return out


def qlike(r2: np.ndarray, v: np.ndarray) -> float:
    ratio = np.maximum(r2, 1e-20) / v
    return float(np.mean(ratio - np.log(ratio) - 1))


def fit_ewma(r_fit: np.ndarray) -> float:
    init = float(np.var(r_fit[:500]))
    grid = np.concatenate([np.arange(0.80, 0.99, 0.01), np.arange(0.99, 0.9995, 0.0015)])
    return float(min(grid, key=lambda lam: qlike(r_fit**2, ewma_var(r_fit, lam, init))))


def fit_garch(r_fit: np.ndarray) -> tuple[float, float, float]:
    vbar = float(np.var(r_fit))
    best, best_ll = (0.05, 0.9), -np.inf
    for a in np.arange(0.01, 0.31, 0.01):
        for b in np.arange(0.50, 0.995, 0.01):
            if a + b >= 0.999:
                continue
            v = garch_var(r_fit, a, b, vbar)
            ll = -0.5 * np.sum(np.log(v) + r_fit**2 / v)
            if ll > best_ll:
                best, best_ll = (float(a), float(b)), ll
    return best[0], best[1], vbar


def fhs_quantiles(cum_var: np.ndarray, z_resid: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """(N, H, Q) quantiles: sqrt(cumulative variance) times empirical quantiles of standardized residuals."""
    zq = np.quantile(z_resid, levels)
    return np.sqrt(cum_var)[:, :, None] * zq[None, None, :]


def baseline_quantiles(logc: np.ndarray, idx: np.ndarray, levels: np.ndarray, fit_end: int) -> dict[str, np.ndarray]:
    """Quantiles of the k-hour log return after each origin, k = 1..H, for every baseline.

    `fit_end` is the bar index where the tuning half ends; parameters only see returns before it.
    """
    r = np.diff(logc, prepend=np.nan)  # r[t] = log(c[t] / c[t-1])
    out: dict[str, np.ndarray] = {}
    z = np.array([NormalDist().inv_cdf(p) for p in levels])
    k = np.arange(1, H + 1)

    # random walk: sigma of the last 500 hourly returns
    sig = np.array([np.nanstd(r[max(1, i - 500) : i]) for i in idx])
    out["rw"] = sig[:, None, None] * np.sqrt(k)[None, :, None] * z[None, None, :]

    # historical simulation: empirical k-hour returns over the last 2000 hours
    hs = np.empty((len(idx), H, len(levels)))
    for n, i in enumerate(idx):
        window = logc[max(0, i - 2000 - H) : i]
        for kk in range(1, H + 1):
            diffs = window[kk:] - window[:-kk]
            hs[n, kk - 1] = np.quantile(diffs[-2000:], levels)
    out["hs"] = hs

    r_all = np.nan_to_num(r, nan=0.0)
    r_fit = r_all[1:fit_end]

    lam = fit_ewma(r_fit)
    v_e = ewma_var(r_all, lam, float(np.var(r_fit[:500])))  # v_e[t]: forecast for r[t]
    z_e = r_fit / np.sqrt(v_e[1:fit_end])
    cum_e = v_e[idx][:, None] * k[None, :]
    out["ewma"] = fhs_quantiles(cum_e, z_e, levels)

    a, b, vbar = fit_garch(r_fit)
    v_g = garch_var(r_all, a, b, vbar)
    z_g = r_fit / np.sqrt(v_g[1:fit_end])
    v1 = v_g[idx]  # one-step forecast for r[i], the first hour after the origin
    steps = vbar + np.power(a + b, k - 1)[None, :] * (v1[:, None] - vbar)
    out["garch"] = fhs_quantiles(np.cumsum(steps, axis=1), z_g, levels)
    out["_params"] = {"ewma_lambda": lam, "garch_alpha": a, "garch_beta": b}
    return out


# ---------------------------------------------------------------- scores


def crps(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """(N, H) CRPS from quantiles q (N, H, Q) and outcomes y (N, H): twice the mean pinball loss."""
    d = y[..., None] - q
    return 2 * np.maximum(levels * d, (levels - 1) * d).mean(axis=-1)


def sigma_from_quantiles(q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """(N, H) volatility implied by the 15th to 85th percentile spread (robust to fat tails)."""
    lo, hi = int(np.argmin(np.abs(levels - 0.15))), int(np.argmin(np.abs(levels - 0.85)))
    zspan = NormalDist().inv_cdf(levels[hi]) - NormalDist().inv_cdf(levels[lo])
    return np.maximum((q[..., hi] - q[..., lo]) / zspan, 1e-9)


def kupiec(hits: np.ndarray, p: float) -> float:
    """p-value of the unconditional coverage (proportion of failures) test."""
    n, x = len(hits), int(hits.sum())
    if x in (0, n):
        ll1 = 0.0
    else:
        ph = x / n
        ll1 = (n - x) * math.log(1 - ph) + x * math.log(ph)
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    lr = -2 * (ll0 - ll1)
    return float(math.erfc(math.sqrt(max(lr, 0) / 2)))  # chi2(1) survival


def christoffersen(hits: np.ndarray) -> float:
    """p-value of the independence test (are VaR breaches clustered?)."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    if n01 + n11 == 0 or n00 + n10 == 0:
        return 1.0

    def ll(p, x0, x1):
        return (x0 * math.log(1 - p) if x0 and p < 1 else 0) + (x1 * math.log(p) if x1 and p > 0 else 0)

    p01, p11 = n01 / max(1, n00 + n01), n11 / max(1, n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(math.erfc(math.sqrt(max(lr, 0) / 2)))


def block_resamples(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = math.ceil(n / block)
    starts = rng.integers(0, max(1, n - block + 1), size=(N_BOOT, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(N_BOOT, -1)[:, :n]


# ---------------------------------------------------------------- volatility targeting


def vol_target(r_next: np.ndarray, sigma24: np.ndarray, target: float, rebalance: np.ndarray) -> dict:
    """Hourly P&L of holding w = min(MAX, target / forecast sigma), changed only at rebalance hours, after fees."""
    w = np.empty(len(r_next))
    cur = 0.0
    turnover = 0.0
    pnl = np.empty(len(r_next))
    for t in range(len(r_next)):
        if rebalance[t]:
            new = min(MAX_LEVERAGE, target / sigma24[t]) if sigma24 is not None else 1.0
            turnover += abs(new - cur)
            pnl[t] = -FEE * abs(new - cur)
            cur = new
        else:
            pnl[t] = 0.0
        w[t] = cur
        pnl[t] += cur * (math.exp(r_next[t]) - 1)
    return {"pnl": pnl, "weight": w, "turnover": turnover}


def perf(pnl: np.ndarray, target_hourly: float | None = None) -> dict:
    eq = np.cumsum(np.log1p(pnl))
    years = len(pnl) / HOURS_PER_YEAR
    vol = float(np.std(pnl) * math.sqrt(HOURS_PER_YEAR))
    ret = float(math.exp(eq[-1] / years) - 1) if years > 0 else 0.0
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    out = {
        "ann_return": ret,
        "ann_vol": vol,
        "sharpe": float(np.mean(pnl) / np.std(pnl) * math.sqrt(HOURS_PER_YEAR)) if np.std(pnl) > 0 else 0.0,
        "max_drawdown": 1 - math.exp(-dd),
    }
    # how steady the risk was: spread of rolling 7-day realized volatility, annualized
    roll = np.array([np.std(pnl[s : s + 168]) for s in range(0, len(pnl) - 168, 24)]) * math.sqrt(HOURS_PER_YEAR)
    out["vol_of_vol"] = float(np.std(roll)) if len(roll) else 0.0
    return out


# ---------------------------------------------------------------- main


def evaluate_pair(symbol: str) -> dict:
    f = np.load(RESULTS / f"{symbol}.npz")
    idx_all, q_t, levels = f["index"], f["q"].astype(np.float64), f["levels"]
    times, close, _ = load_klines(symbol, "1h", BARS)
    logc = np.log(close)
    keep = idx_all + H <= len(close)
    idx, q_t = idx_all[keep], q_t[keep]
    half = len(idx) // 2
    fit_end = int(idx[half])  # baselines are fitted on returns before the held-out half
    base = baseline_quantiles(logc, idx, levels, fit_end)
    params = base.pop("_params")
    quant = {"timestral": q_t, **base}
    quant["blend"] = 0.5 * (q_t + base["garch"])
    y = np.stack([logc[i : i + H] - logc[i - 1] for i in idx])  # (N, H) k-hour log returns
    r_hourly = np.diff(logc)  # r_hourly[t] = log(c[t+1]/c[t])
    realized_var = np.cumsum(np.stack([r_hourly[i - 1 : i - 1 + H] ** 2 for i in idx]), axis=1)

    out = {"params": params, "periods": {}}
    for period, sl in (("tuning", slice(0, half)), ("heldout", slice(half, None))):
        res: dict = {"n": int(len(idx[sl]))}
        yy = y[sl]
        crps_by = {m: crps(quant[m][sl], yy, levels) for m in MODELS}
        res["crps"] = {m: {k: float(crps_by[m][:, k - 1].mean()) for k in STEPS} | {"avg": float(crps_by[m].mean())} for m in MODELS}
        res["crps_rows"] = {m: crps_by[m].mean(axis=1) for m in MODELS}  # per origin, for the bootstrap
        res["crps_steps"] = {m: crps_by[m].sum(axis=0) for m in MODELS}  # (H,) sums, for the per-hour curves
        res["dates"] = [int(times[idx[sl][0]]), int(times[idx[sl][-1]])]
        sig = {m: sigma_from_quantiles(quant[m][sl], levels) for m in MODELS}
        res["qlike"] = {m: {k: qlike(realized_var[sl][:, k - 1], sig[m][:, k - 1] ** 2 * 1.0) for k in STEPS} for m in MODELS}
        # Value-at-Risk on the left tail (long positions)
        var = {}
        for m in MODELS:
            var[m] = {}
            for p in (0.01, 0.05):
                li = int(np.argmin(np.abs(levels - p)))
                for k in (1, 24):
                    sel = np.arange(0, len(yy), k)  # non-overlapping windows
                    hits = yy[sel, k - 1] < quant[m][sl][sel, k - 1, li]
                    d = yy[sel, k - 1] - quant[m][sl][sel, k - 1, li]
                    var[m][f"{int(p * 100)}%@{k}h"] = {
                        "hit_rate": float(hits.mean()),
                        "kupiec_p": kupiec(hits, p),
                        "christoffersen_p": christoffersen(hits) if k == 1 else None,
                        "qloss": float(np.mean(np.maximum(p * d, (p - 1) * d))),
                    }
        res["var"] = var
        # volatility targeting, rebalanced every 24 hours at origin hours
        origin_idx = idx[sl]
        r_next = logc[origin_idx] - logc[origin_idx - 1]  # return of the hour that starts at each origin
        rebalance = (np.arange(len(origin_idx)) % 24) == 0
        hist_r = np.diff(logc[: fit_end])
        target = float(np.median([np.std(hist_r[s : s + 24]) for s in range(0, len(hist_r) - 24, 24)]) * math.sqrt(24))
        strat = {"buy_and_hold": vol_target(r_next, None, target, rebalance)}
        for m in MODELS:
            strat[m] = vol_target(r_next, sig[m][:, H - 1], target, rebalance)
        res["vt"] = {name: perf(s["pnl"]) | {"turnover": s["turnover"], "avg_weight": float(s["weight"].mean())} for name, s in strat.items()}
        res["vt_pnl"] = {name: s["pnl"] for name, s in strat.items()}
        res["target_ann_vol"] = target * math.sqrt(HOURS_PER_YEAR / 24)
        out["periods"][period] = res
    return out


def main() -> None:
    pairs = [p for p in CONFIRM_PAIRS if (RESULTS / f"{p}.npz").exists()]
    per_pair = {p: evaluate_pair(p) for p in pairs}
    rng = np.random.default_rng(0)
    lines = [f"Pairs: {', '.join(pairs)}. Baseline parameters fitted on the tuning half only.", ""]
    summary: dict = {"pairs": pairs, "params": {p: v["params"] for p, v in per_pair.items()}, "periods": {}}
    for period in ("tuning", "heldout"):
        P = {p: per_pair[p]["periods"][period] for p in pairs}
        S: dict = {}
        # 1. pooled CRPS skill of Timestral vs each baseline, with paired block-bootstrap CIs
        lines += [f"## {period}", "", "### Forecast quality", "",
                  "CRPS skill of a model vs a baseline, % (90% CI); QLIKE of the 24 h volatility, lower is better", "",
                  "| model vs baseline | h=1 | h=8 | h=24 | avg 1-24 (90% CI) | QLIKE 24h baseline | QLIKE 24h model |", "|---|---|---|---|---|---|---|"]
        S["crps_skill"] = {}
        for model, m in (("timestral", "rw"), ("timestral", "hs"), ("timestral", "ewma"), ("timestral", "garch"), ("blend", "garch")):
            cells = []
            for k in STEPS:
                num = sum(P[p]["crps"][model][k] for p in pairs)
                den = sum(P[p]["crps"][m][k] for p in pairs)
                cells.append(1 - num / den)
            num_rows = {p: P[p]["crps_rows"][model] for p in pairs}
            den_rows = {p: P[p]["crps_rows"][m] for p in pairs}
            boots = np.zeros(N_BOOT)
            nsum = np.zeros(N_BOOT)
            dsum = np.zeros(N_BOOT)
            for p in pairs:
                ix = block_resamples(len(num_rows[p]), 48, rng)
                nsum += num_rows[p][ix].mean(axis=1)
                dsum += den_rows[p][ix].mean(axis=1)
            boots = 1 - nsum / dsum
            avg = 1 - sum(r.mean() for r in num_rows.values()) / sum(r.mean() for r in den_rows.values())
            ql_b = np.mean([P[p]["qlike"][m][24] for p in pairs])
            ql_t = np.mean([P[p]["qlike"][model][24] for p in pairs])
            S["crps_skill"][f"{model}_vs_{m}"] = {"steps": dict(zip(STEPS, cells)), "avg": avg, "ci": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))], "qlike_base": ql_b, "qlike_timestral": ql_t}
            lines.append(f"| {model} vs {m} | {cells[0] * 100:+.2f} | {cells[1] * 100:+.2f} | {cells[2] * 100:+.2f} | {avg * 100:+.2f} ({np.percentile(boots, 5) * 100:+.2f}, {np.percentile(boots, 95) * 100:+.2f}) | {ql_b:.4f} | {ql_t:.4f} |")
        # 2. VaR
        lines += ["", "### Value-at-Risk (left tail, pooled over pairs)", "",
                  "Hit rate should match the level; Kupiec p < 0.05 rejects the model; quantile loss: lower is better.", "",
                  "| model | 1%@1h hits | Kupiec p | Chr. p | qloss | 5%@1h hits | Kupiec p | 1%@24h hits | 5%@24h hits | qloss 5%@24h |", "|---|---|---|---|---|---|---|---|---|---|"]
        S["var"] = {}
        for m in MODELS:
            row = {}
            for key in ("1%@1h", "5%@1h", "1%@24h", "5%@24h"):
                hr = np.mean([P[p]["var"][m][key]["hit_rate"] for p in pairs])
                kp = float(np.median([P[p]["var"][m][key]["kupiec_p"] for p in pairs]))
                ql = float(np.mean([P[p]["var"][m][key]["qloss"] for p in pairs]))
                cp = [P[p]["var"][m][key]["christoffersen_p"] for p in pairs]
                cp = float(np.median(cp)) if cp[0] is not None else None
                row[key] = {"hit_rate": hr, "kupiec_p_median": kp, "christoffersen_p_median": cp, "qloss": ql,
                            "kupiec_rejections": int(sum(P[p]["var"][m][key]["kupiec_p"] < 0.05 for p in pairs))}
            S["var"][m] = row
            r1, r5, r124, r524 = row["1%@1h"], row["5%@1h"], row["1%@24h"], row["5%@24h"]
            npairs = len(pairs)
            lines.append(f"| {m} | {r1['hit_rate'] * 100:.2f}% | {r1['kupiec_p_median']:.2f} ({r1['kupiec_rejections']}/{npairs} rej.) | {r1['christoffersen_p_median']:.2f} | {r1['qloss'] * 1e4:.3f} | {r5['hit_rate'] * 100:.2f}% | {r5['kupiec_p_median']:.2f} ({r5['kupiec_rejections']}/{npairs}) | {r124['hit_rate'] * 100:.2f}% | {r524['hit_rate'] * 100:.2f}% | {r524['qloss'] * 1e4:.2f} |")
        # 3. volatility targeting: equal-weight basket of the per-pair strategies
        lines += ["", "### Volatility targeting (equal-weight basket of the eight pairs, daily resizing, 10 bp fees)", "",
                  "| sizing | ann. return | ann. vol | Sharpe (90% CI) | max drawdown | vol of vol | turnover/pair |", "|---|---|---|---|---|---|---|"]
        n = min(len(P[p]["vt_pnl"]["buy_and_hold"]) for p in pairs)
        S["vt"] = {}
        basket = {name: np.mean([P[p]["vt_pnl"][name][-n:] for p in pairs], axis=0) for name in ("buy_and_hold",) + MODELS}
        ix = block_resamples(n, 24 * 7, rng)
        for name, pnl in basket.items():
            pf = perf(pnl)
            sh_boot = pnl[ix].mean(axis=1) / pnl[ix].std(axis=1) * math.sqrt(HOURS_PER_YEAR)
            turn = np.mean([P[p]["vt"][name]["turnover"] for p in pairs])
            S["vt"][name] = pf | {"sharpe_ci": [float(np.percentile(sh_boot, 5)), float(np.percentile(sh_boot, 95))], "turnover": turn}
            lines.append(f"| {name} | {pf['ann_return'] * 100:+.1f}% | {pf['ann_vol'] * 100:.1f}% | {pf['sharpe']:.2f} ({np.percentile(sh_boot, 5):.2f}, {np.percentile(sh_boot, 95):.2f}) | {pf['max_drawdown'] * 100:.1f}% | {pf['vol_of_vol'] * 100:.1f}% | {turn:.1f} |")
        # paired Sharpe differences of the sizing rules (same days resampled for both)
        lines.append("")
        for a_name, b_name in (("timestral", "garch"), ("blend", "garch"), ("timestral", "buy_and_hold")):
            ta, tb = basket[a_name], basket[b_name]
            d = (ta[ix].mean(axis=1) / ta[ix].std(axis=1) - tb[ix].mean(axis=1) / tb[ix].std(axis=1)) * math.sqrt(HOURS_PER_YEAR)
            key = f"sharpe_diff_{a_name}_vs_{b_name}"
            S["vt"][key] = {"point": S["vt"][a_name]["sharpe"] - S["vt"][b_name]["sharpe"],
                            "ci": [float(np.percentile(d, 5)), float(np.percentile(d, 95))]}
            lines.append(f"Sharpe difference, {a_name} vs {b_name} sizing: {S['vt'][key]['point']:+.2f} "
                         f"(90% CI {np.percentile(d, 5):+.2f}, {np.percentile(d, 95):+.2f})  ")
        lines.append("")
        # per-hour CRPS skill curves vs GARCH, and the basket equity curves, for the figures
        S["curves"] = {
            model: [float(1 - sum(P[p]["crps_steps"][model][k] for p in pairs) / sum(P[p]["crps_steps"]["garch"][k] for p in pairs)) for k in range(H)]
            for model in ("timestral", "blend", "ewma", "hs", "rw")
        }
        S["equity"] = {name: np.cumsum(np.log1p(basket[name]))[::24].tolist() for name in ("buy_and_hold", "timestral", "garch", "blend")}
        S["dates"] = P[pairs[0]]["dates"]
        summary["periods"][period] = S
    text = "\n".join(lines)
    (RESULTS / "summary.md").write_text(text + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    main()
