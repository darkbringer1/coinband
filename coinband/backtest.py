"""Walk-forward backtest of a quantile forecaster on one price series.

At each forecast origin `i` the forecaster sees only `close[:i]` (at most `context` bars of it) and
forecasts `close[i : i + horizon]`. Origins are spaced `stride` bars apart and anchored on the data
end, so the last origin is `i = len(close)`: the live forecast past the last closed candle.

All origins are batched into chunked `forecaster.forecast` calls. Contexts shorter than `context`
(early origins) are left-padded with NaN, which Timestral treats as missing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

BAND_LEVELS = (0.05, 0.1, 0.5, 0.9, 0.95)
LO, HI = 0.1, 0.9  # the scored 80% band


@dataclass(frozen=True)
class BacktestParams:
    context: int = 2048
    horizon: int = 24
    stride: int = 24
    max_origins: int = 400
    min_context: int = 64
    batch_size: int = 32

    def validate(self, n_bars: int) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.context < 1:
            raise ValueError("context must be >= 1")
        if self.max_origins < 1:
            raise ValueError("max_origins must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        need = max(1, min(self.min_context, self.context))
        if n_bars < need:
            raise ValueError(f"need at least {need} bars, got {n_bars}")


def origin_indices(n_bars: int, params: BacktestParams) -> list[int]:
    """Origins, oldest first, ending at `n_bars` (the live origin), `stride` apart."""
    first = max(1, min(params.min_context, params.context))
    origins = []
    i = n_bars
    while i >= first and len(origins) < params.max_origins:
        origins.append(i)
        i -= params.stride
    return origins[::-1]


def _level_indices(levels: Sequence[float], wanted: Sequence[float]) -> list[int]:
    out = []
    for w in wanted:
        best = min(range(len(levels)), key=lambda k: abs(levels[k] - w))
        if abs(levels[best] - w) > 1e-6:
            raise ValueError(f"forecaster has no {w} quantile (levels: {tuple(levels)})")
        out.append(best)
    return out


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def score_origin(
    close: np.ndarray, origin: int, bands: np.ndarray, horizon: int, lo: int, mid: int, hi: int
) -> dict[str, float | int | bool | None]:
    """Scores of one origin over the bars that have been realized.

    `bands` is `(H, len(BAND_LEVELS))`. Returns the realized step count, the 80% coverage, the mean
    absolute error of the median and of the naive last-value forecast, and whether the median got the
    direction at the horizon end right (only once the whole horizon is realized; a flat move counts as
    a miss for both up and down calls).
    """
    realized = close[origin : origin + horizon]
    k = len(realized)
    out: dict[str, float | int | bool | None] = {
        "realized": k,
        "coverage": None,
        "mae": None,
        "naive_mae": None,
        "direction_hit": None,
    }
    if k == 0:
        return out
    last = close[origin - 1]
    b = bands[:k]
    inside = (realized >= b[:, lo]) & (realized <= b[:, hi])
    out["coverage"] = float(inside.mean())
    out["mae"] = float(np.abs(b[:, mid] - realized).mean())
    out["naive_mae"] = float(np.abs(last - realized).mean())
    if k == horizon:
        pred = np.sign(bands[horizon - 1, mid] - last)
        real = np.sign(realized[-1] - last)
        out["direction_hit"] = bool(pred != 0 and pred == real)
    return out


def summarize(origins: list[dict]) -> dict[str, float | int | None]:
    """Aggregate scores over fully realized origins."""
    done = [o for o in origins if o["direction_hit"] is not None]
    if not done:
        return {"scored": 0, "coverage": None, "mae": None, "naive_mae": None, "skill": None, "direction_hit_rate": None}
    mae = sum(o["mae"] for o in done) / len(done)
    naive = sum(o["naive_mae"] for o in done) / len(done)
    return {
        "scored": len(done),
        "coverage": sum(o["coverage"] for o in done) / len(done),
        "mae": mae,
        "naive_mae": naive,
        # > 0 means the median beats "price stays flat".
        "skill": (1.0 - mae / naive) if naive > 0 else None,
        "direction_hit_rate": sum(o["direction_hit"] for o in done) / len(done),
    }


def run_backtest(
    forecaster,
    close: Sequence[float],
    params: BacktestParams,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Walk-forward backtest; see the module docstring.

    `forecaster` needs `quantile_levels` and `forecast(context (B, 1, T), prediction_length)` returning
    an object with `quantiles` `(B, 1, H, Q)`. `progress(done, total)` is called after each chunk.

    Returns `{"levels", "origins": [{"index", "bands": (H, 5) list, scores...}], "summary"}`.
    """
    series = np.asarray(close, dtype=np.float64)
    n = len(series)
    params.validate(n)
    origins = origin_indices(n, params)
    level_idx = _level_indices(forecaster.quantile_levels, BAND_LEVELS)
    lo, mid, hi = BAND_LEVELS.index(LO), BAND_LEVELS.index(0.5), BAND_LEVELS.index(HI)

    ctx_len = min(params.context, max(origins))
    results: list[dict] = []
    total = len(origins)
    if progress is not None:
        progress(0, total)
    for start in range(0, total, params.batch_size):
        chunk = origins[start : start + params.batch_size]
        batch = np.full((len(chunk), 1, ctx_len), np.nan, dtype=np.float32)
        for row, i in enumerate(chunk):
            window = series[max(0, i - ctx_len) : i]
            batch[row, 0, ctx_len - len(window) :] = window
        forecast = forecaster.forecast(batch, prediction_length=params.horizon)
        q = _to_numpy(forecast.quantiles)
        if q.shape[:3] != (len(chunk), 1, params.horizon):
            raise ValueError(f"forecaster returned quantiles shaped {q.shape}")
        q = q[:, 0][..., level_idx]  # (B, H, 5)
        for row, i in enumerate(chunk):
            bands = q[row]
            entry = {"index": i, "bands": bands}
            entry.update(score_origin(series, i, bands, params.horizon, lo, mid, hi))
            results.append(entry)
        if progress is not None:
            progress(len(results), total)

    summary = summarize(results)
    for entry in results:
        entry["bands"] = [[_round(v) for v in step] for step in entry["bands"]]
    return {"levels": list(BAND_LEVELS), "origins": results, "summary": summary}


def _round(x: float, digits: int = 7) -> float | None:
    """Round to `digits` significant digits, keeping JSON small; non-finite becomes None."""
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")
