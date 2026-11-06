"""Phase 1b follow-up: is the cross-sectional skew effect more than momentum or volatility in disguise?

At each hour, rank-regress Timestral's 24 h band skew across the eight pairs on the past 24 h return and the
forecast band width, and test what is left (the residual) as a reversed long-short signal, against the raw
reversed skew, plain 24 h reversal, and a low-volatility signal. Same halves, fees and bootstrap as
`direction.py`; also with 20 bp fees and 8-hour rebalancing as robustness checks.

    uv run python experiments/phase1/skew_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction import CONFIRM_PAIRS, HOURS_PER_YEAR, RESULTS, blocks, load_pair, rankdata  # noqa: E402

N_BOOT = 1000


def zrank(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, axis=1)
    r -= r.mean(axis=1, keepdims=True)
    return r / np.maximum(r.std(axis=1, keepdims=True), 1e-12)


def residualize(y: np.ndarray, xs: list[np.ndarray]) -> np.ndarray:
    """Per row (hour), the residual of y on the columns of xs (all cross-sectional ranks), by least squares."""
    out = np.empty_like(y)
    for t in range(len(y)):
        A = np.column_stack([np.ones(y.shape[1])] + [x[t] for x in xs])
        beta, *_ = np.linalg.lstsq(A, y[t], rcond=None)
        out[t] = y[t] - A @ beta
    return out


def long_short(X: np.ndarray, r_hour: np.ndarray, fee: float, every: int) -> np.ndarray:
    m, P = X.shape
    w = np.zeros(P)
    pnl = np.zeros(m)
    for t in range(m):
        if t % every == 0:
            order = np.argsort(X[t])
            new = np.zeros(P)
            new[order[-2:]] = 0.5
            new[order[:2]] = -0.5
            pnl[t] -= fee * np.abs(new - w).sum()
            w = new
        pnl[t] += (w * r_hour[t]).sum()
    return pnl


def main() -> None:
    data = {p: load_pair(p) for p in CONFIRM_PAIRS}
    pairs = list(data)
    n = min(len(d["idx"]) for d in data.values())
    half = n // 2
    rng = np.random.default_rng(1)
    f = {p: np.load(RESULTS / f"{p}.npz") for p in pairs}
    lines = []
    for period, sl in (("tuning", slice(0, half)), ("heldout", slice(half, n))):
        m = sl.stop - sl.start
        get = lambda name: np.stack([data[p]["sig"][(name, 24)][sl] for p in pairs], axis=1)  # noqa: E731
        skew, median, mom = get("skew"), get("median"), get("mom24")
        lv = f[pairs[0]]["levels"]
        i10, i90 = int(np.argmin(np.abs(lv - 0.1))), int(np.argmin(np.abs(lv - 0.9)))
        width = np.stack([(f[p]["q"][:n, 23, i90] - f[p]["q"][:n, 23, i10])[sl] for p in pairs], axis=1)
        r_hour = np.stack([np.exp(np.diff(data[p]["logc"])[data[p]["idx"][sl] - 1]) - 1 for p in pairs], axis=1)
        Y = np.stack([data[p]["y"][24][sl] for p in pairs], axis=1)
        # how much do the median and skew track volatility (band width) across pairs?
        c_med = np.mean([np.corrcoef(zrank(median)[t], zrank(width)[t])[0, 1] for t in range(0, m, 24)])
        c_skw = np.mean([np.corrcoef(zrank(skew)[t], zrank(width)[t])[0, 1] for t in range(0, m, 24)])
        resid = residualize(zrank(skew), [zrank(mom), zrank(width)])
        signals = {
            "-skew (reversed raw skew)": -skew,
            "-skew residual (after momentum and width)": -resid,
            "rev24 (minus past 24 h return)": -mom,
            "low volatility (minus band width)": -width,
        }
        lines += [f"## {period}", "",
                  f"Mean cross-sectional rank correlation with band width: median {c_med:+.2f}, skew {c_skw:+.2f}", "",
                  "| signal | xs IC (90% CI) | Sharpe 10 bp, daily (90% CI) | ann. return | Sharpe 20 bp | Sharpe 10 bp, every 8 h |",
                  "|---|---|---|---|---|---|"]
        sb = blocks(m, 24 * 7, rng)
        for name, X in signals.items():
            rx, ry = zrank(X), zrank(Y)
            ic_t = (rx * ry).mean(axis=1)
            ib = blocks(m, 48, rng)
            ic_b = ic_t[ib].mean(axis=1)
            res = {}
            for key, fee, every in (("base", 0.001, 24), ("fee20", 0.002, 24), ("8h", 0.001, 8)):
                pnl = long_short(X, r_hour, fee, every)
                sh = pnl.mean() / pnl.std() * math.sqrt(HOURS_PER_YEAR)
                shb = pnl[sb].mean(axis=1) / pnl[sb].std(axis=1) * math.sqrt(HOURS_PER_YEAR)
                res[key] = (sh, np.percentile(shb, 5), np.percentile(shb, 95), math.exp(np.log1p(pnl).sum() * HOURS_PER_YEAR / m) - 1)
            b = res["base"]
            lines.append(f"| {name} | {ic_t.mean():+.3f} ({np.percentile(ic_b, 5):+.3f}, {np.percentile(ic_b, 95):+.3f}) | "
                         f"{b[0]:+.2f} ({b[1]:+.2f}, {b[2]:+.2f}) | {b[3] * 100:+.1f}% | {res['fee20'][0]:+.2f} | {res['8h'][0]:+.2f} |")
        lines.append("")
    text = "\n".join(lines)
    (RESULTS / "skew_check.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
