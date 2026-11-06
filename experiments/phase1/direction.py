"""Phase 1b: does Timestral carry any directional signal, once measured properly?

Uses the hourly forecasts from `collect.py`. Signals, each computed at every hourly origin for a horizon h:

- `median`: the forecast median log return.
- `skew`: band asymmetry, ((P90 - P50) - (P50 - P10)) / (P90 - P10). Positive: more room up than down.
- `p_up`: the forecast probability that the price ends higher, read off the 21 quantiles.
- Baselines from past prices only: `mom24` (last 24 h return), `rev1` (minus the last 1 h return), `rev24`
  (minus the last 24 h return).

Tests, separately on the tuning and held-out halves:

1. Time-series information coefficient (IC): Spearman correlation between a signal and the realized h-hour
   return, per pair, averaged over pairs, with a block-bootstrap interval.
2. Cross-sectional IC: at each hour, Spearman correlation across the eight pairs between the signal and the
   realized return, averaged over hours.
3. A long-short basket: every 24 hours, long the two pairs with the highest signal and short the two lowest,
   held 24 hours, 10 bp per unit of position change; annualized Sharpe with a block-bootstrap interval.

Many signals and horizons are tested, so a single "significant" cell means little. What counts is a signal that
is positive in the tuning half and stays positive in the held-out half.

    uv run python experiments/phase1/direction.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sweep import CONFIRM_PAIRS, load_klines  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "phase1"
BARS, H = 13152, 24
HORIZONS = (1, 8, 24)
SIGNALS = ("median", "skew", "p_up", "mom24", "rev1", "rev24")
FEE = 0.001
N_BOOT = 1000
HOURS_PER_YEAR = 24 * 365


def rankdata(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.argsort(np.argsort(x, axis=axis), axis=axis).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def p_up(q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """P(return > 0) by linear interpolation of the quantile function at 0, per row; q is (..., Q) increasing."""
    flat = q.reshape(-1, q.shape[-1])
    out = np.empty(len(flat))
    for i, row in enumerate(flat):
        out[i] = 1 - np.interp(0.0, row, levels, left=0.0, right=1.0)
    return out.reshape(q.shape[:-1])


def load_pair(symbol: str) -> dict:
    f = np.load(RESULTS / f"{symbol}.npz")
    idx, q, levels = f["index"], f["q"].astype(np.float64), f["levels"]
    _, close, _ = load_klines(symbol, "1h", BARS)
    logc = np.log(close)
    keep = idx + H <= len(close)
    idx, q = idx[keep], q[keep]
    i10, i50, i90 = (int(np.argmin(np.abs(levels - v))) for v in (0.1, 0.5, 0.9))
    sig = {}
    for h in HORIZONS:
        qh = q[:, h - 1]
        width = np.maximum(qh[:, i90] - qh[:, i10], 1e-12)
        sig[("median", h)] = qh[:, i50]
        sig[("skew", h)] = ((qh[:, i90] - qh[:, i50]) - (qh[:, i50] - qh[:, i10])) / width
        sig[("p_up", h)] = p_up(qh, levels)
    past1 = logc[idx - 1] - logc[idx - 2]
    past24 = logc[idx - 1] - logc[idx - 25]
    for h in HORIZONS:
        sig[("mom24", h)] = past24
        sig[("rev1", h)] = -past1
        sig[("rev24", h)] = -past24
    y = {h: logc[idx + h - 1] - logc[idx - 1] for h in HORIZONS}
    return {"idx": idx, "sig": sig, "y": y, "logc": logc}


def blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = math.ceil(n / block)
    starts = rng.integers(0, max(1, n - block + 1), size=(N_BOOT, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(N_BOOT, -1)[:, :n]


def main() -> None:
    data = {p: load_pair(p) for p in CONFIRM_PAIRS if (RESULTS / f"{p}.npz").exists()}
    pairs = list(data)
    n = min(len(d["idx"]) for d in data.values())
    for d in data.values():  # align on the common origins (same bars for every pair)
        assert np.array_equal(d["idx"][:n], data[pairs[0]]["idx"][:n])
    half = n // 2
    rng = np.random.default_rng(0)
    lines = [f"Pairs: {', '.join(pairs)}. Origins per pair: {n} (hourly), split in half.", ""]
    out: dict = {}
    for period, sl in (("tuning", slice(0, half)), ("heldout", slice(half, n))):
        m = sl.stop - sl.start
        res: dict = {}
        lines += [f"## {period}", "", "### Time-series IC (Spearman, mean over pairs, 90% CI) and sign hit rate", "",
                  "| signal | " + " | ".join(f"IC h={h}" for h in HORIZONS) + " | " + " | ".join(f"hit h={h}" for h in HORIZONS) + " |",
                  "|---|" + "---|" * (2 * len(HORIZONS))]
        ix = blocks(m, 48, rng)
        for s in SIGNALS:
            cells_ic, cells_hit = [], []
            for h in HORIZONS:
                ics, boots, hits = [], np.zeros(N_BOOT), []
                for p in pairs:
                    x, yy = data[p]["sig"][(s, h)][sl], data[p]["y"][h][sl]
                    ics.append(spearman(x, yy))
                    hits.append(float(np.mean(np.sign(x) == np.sign(yy))) if s != "p_up" else float(np.mean((x > 0.5) == (yy > 0))))
                    rx, ry = rankdata(x[ix], axis=1), rankdata(yy[ix], axis=1)
                    rx -= rx.mean(axis=1, keepdims=True)
                    ry -= ry.mean(axis=1, keepdims=True)
                    boots += (rx * ry).sum(axis=1) / np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
                boots /= len(pairs)
                ic = float(np.mean(ics))
                lo, hi = np.percentile(boots, 5), np.percentile(boots, 95)
                res[f"ts_ic_{s}_{h}"] = {"ic": ic, "ci": [float(lo), float(hi)], "hit": float(np.mean(hits))}
                star = "**" if lo > 0 or hi < 0 else ""
                cells_ic.append(f"{star}{ic:+.3f} ({lo:+.3f}, {hi:+.3f}){star}")
                cells_hit.append(f"{np.mean(hits) * 100:.1f}%")
            lines.append(f"| {s} | " + " | ".join(cells_ic) + " | " + " | ".join(cells_hit) + " |")

        # cross-sectional IC and a long-short basket at h = 24
        lines += ["", "### Cross-sectional (eight pairs ranked each hour), h = 24", "",
                  "| signal | mean cross-sectional IC (90% CI) | long-short Sharpe (90% CI) | ann. return | turnover/day |", "|---|---|---|---|---|"]
        h = 24
        Y = np.stack([data[p]["y"][h][sl] for p in pairs], axis=1)  # (m, P)
        r_hour = np.stack([np.exp(np.diff(data[p]["logc"])[data[p]["idx"][sl] - 1]) - 1 for p in pairs], axis=1)  # (m, P) next-hour returns
        for s in SIGNALS:
            X = np.stack([data[p]["sig"][(s, h)][sl] for p in pairs], axis=1)
            rx, ry = rankdata(X, axis=1), rankdata(Y, axis=1)
            rx -= rx.mean(axis=1, keepdims=True)
            ry -= ry.mean(axis=1, keepdims=True)
            ic_t = (rx * ry).sum(axis=1) / np.sqrt(np.maximum((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1), 1e-12))
            ib = blocks(m, 48, rng)
            ic_b = ic_t[ib].mean(axis=1)
            # long top 2, short bottom 2, rebalanced every 24 h, equal weights summing to 1 on each side
            w = np.zeros(len(pairs))
            pnl = np.zeros(m)
            turnover = 0.0
            for t in range(m):
                if t % 24 == 0:
                    order = np.argsort(X[t])
                    new = np.zeros(len(pairs))
                    new[order[-2:]] = 0.5
                    new[order[:2]] = -0.5
                    cost = FEE * np.abs(new - w).sum()
                    turnover += np.abs(new - w).sum()
                    w = new
                    pnl[t] -= cost
                pnl[t] += (w * r_hour[t]).sum()
            sb = blocks(m, 24 * 7, rng)
            sharpe = float(pnl.mean() / pnl.std() * math.sqrt(HOURS_PER_YEAR)) if pnl.std() > 0 else 0.0
            sh_b = pnl[sb].mean(axis=1) / pnl[sb].std(axis=1) * math.sqrt(HOURS_PER_YEAR)
            ann = float(math.exp(np.log1p(pnl).sum() * HOURS_PER_YEAR / m) - 1)
            res[f"xs_{s}"] = {"ic": float(ic_t.mean()), "ic_ci": [float(np.percentile(ic_b, 5)), float(np.percentile(ic_b, 95))],
                              "sharpe": sharpe, "sharpe_ci": [float(np.percentile(sh_b, 5)), float(np.percentile(sh_b, 95))], "ann_return": ann}
            lines.append(f"| {s} | {ic_t.mean():+.3f} ({np.percentile(ic_b, 5):+.3f}, {np.percentile(ic_b, 95):+.3f}) | "
                         f"{sharpe:+.2f} ({np.percentile(sh_b, 5):+.2f}, {np.percentile(sh_b, 95):+.2f}) | {ann * 100:+.1f}% | {turnover / (m / 24):.2f} |")
        # how much of the Timestral signals is just past returns?
        lines += ["", "### Overlap with simple signals (time-series Spearman between signals, mean over pairs), h = 24", "",
                  "| | mom24 | rev1 |", "|---|---|---|"]
        for s in ("median", "skew", "p_up"):
            c = [np.mean([spearman(data[p]["sig"][(s, 24)][sl], data[p]["sig"][(b, 24)][sl]) for p in pairs]) for b in ("mom24", "rev1")]
            res[f"overlap_{s}"] = {"mom24": float(c[0]), "rev1": float(c[1])}
            lines.append(f"| {s} | {c[0]:+.2f} | {c[1]:+.2f} |")
        lines.append("")
        out[period] = res
    text = "\n".join(lines)
    (RESULTS / "direction.md").write_text(text + "\n")
    (RESULTS / "direction.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
