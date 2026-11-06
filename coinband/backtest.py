"""Walk-forward backtest of a quantile forecaster on one price series.

At each forecast origin `i` the forecaster sees only `close[:i]` (at most `context` bars of it) and
forecasts `close[i : i + horizon]`. Origins are spaced `stride` bars apart and anchored on the data
end, so the last origin is `i = len(close)`: the live forecast past the last closed candle.

All origins are batched into chunked `forecaster.forecast` calls. Contexts shorter than `context`
(early origins) are left-padded with NaN, which Timestral treats as missing.

Optional trend: with `trend_window > 0` every forecast is tilted by the drift of the recent price,
the least-squares slope `g` of log price over the last `trend_window` bars: quantile `q` at step `k`
becomes `q * exp(g * k)`. That is a separate, explicit model (Timestral + momentum drift); the plain
Timestral forecast is scored alongside it so the two can be compared.

Optional covariates: with `covariates=True` and a `volume` series, every context also carries two
past-only covariate rows, log(1 + volume) and the absolute 1-bar log return. On 1h BTC, ETH and SOL
this lifted CRPS skill and made the median beat "price stays flat"; it costs about 6x the inference.

Scores, all on what has been realized:

- 80% coverage: share of realized bars inside P10..P90.
- MAE of the median vs the naive "price stays flat" forecast.
- Direction at the horizon end.
- CRPS (approximated as twice the mean pinball loss over the model's quantile levels) vs a naive
  random-walk band: last price times `exp(sigma * sqrt(k) * z)`, sigma from recent 1-bar log returns.
- Per-step versions of coverage and the two skills (accuracy by bars ahead) and calibration: for each
  quantile level, how often the realized price came in at or below it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

BAND_LEVELS = (0.05, 0.1, 0.5, 0.9, 0.95)
LO, HI = 0.1, 0.9  # the scored 80% band


@dataclass(frozen=True)
class BacktestParams:
    context: int = 4096
    horizon: int = 24
    stride: int = 1
    max_origins: int = 1000
    min_context: int = 64
    batch_size: int = 64
    trend_window: int = 0  # 0: plain forecast; N: add the drift of the last N bars
    vol_window: int = 500  # bars of 1-bar returns behind the naive random-walk band
    covariates: bool = False  # feed log volume and |1-bar log return| as past covariates

    def validate(self, n_bars: int) -> None:
        for name in ("horizon", "stride", "context", "max_origins", "batch_size", "vol_window"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.trend_window < 0 or self.trend_window == 1:
            raise ValueError("trend_window must be 0 (off) or >= 2")
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


def covariate_rows(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """`(2, T)` past covariates for the whole series: log(1 + volume) and |1-bar log return| (NaN first)."""
    abs_ret = np.abs(np.diff(np.log(close), prepend=np.nan))
    return np.stack([np.log1p(np.maximum(volume, 0.0)), abs_ret]).astype(np.float32)


def trend_slope(close: np.ndarray, origin: int, window: int) -> float:
    """Least-squares slope of log price per bar over the `window` bars before `origin`."""
    y = np.log(close[max(0, origin - window) : origin])
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def apply_trend(q: np.ndarray, slope: float) -> np.ndarray:
    """Tilt `(H, Q)` quantiles by a log-price drift of `slope` per bar."""
    steps = np.arange(1, q.shape[0] + 1, dtype=np.float64)
    return q * np.exp(slope * steps)[:, None]


def naive_quantiles(close: np.ndarray, origin: int, horizon: int, levels: Sequence[float], vol_window: int) -> np.ndarray:
    """`(H, Q)` random-walk band around the last price, from the recent 1-bar log-return volatility."""
    window = close[max(0, origin - vol_window - 1) : origin]
    r = np.diff(np.log(window))
    r = r[np.isfinite(r)]
    sigma = float(r.std()) if len(r) >= 2 else 0.0
    z = np.array([NormalDist().inv_cdf(p) for p in levels])
    steps = np.sqrt(np.arange(1, horizon + 1, dtype=np.float64))
    return close[origin - 1] * np.exp(sigma * steps[:, None] * z[None, :])


def _crps_steps(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per-step CRPS approximation: twice the mean pinball loss over the quantile levels."""
    diff = y[:, None] - q
    return 2.0 * np.maximum(levels * diff, (levels - 1.0) * diff).mean(axis=1)


def score_origin(
    close: np.ndarray,
    origin: int,
    q: np.ndarray,
    levels: Sequence[float],
    horizon: int,
    naive_q: np.ndarray | None = None,
) -> dict:
    """Scores of one origin over its realized bars. `q` and `naive_q` are `(H, Q)` at `levels`.

    Returns the realized step count, 80% coverage, MAE of the median and of "price stays flat", CRPS
    of the forecast and of the naive band (when given), and whether the median got the direction at
    the horizon end right (only once the whole horizon is realized; a flat call or move is a miss).
    """
    lo, mid, hi = _level_indices(levels, (LO, 0.5, HI))
    realized = close[origin : origin + horizon]
    k = len(realized)
    out: dict = {
        "realized": k,
        "coverage": None,
        "mae": None,
        "naive_mae": None,
        "crps": None,
        "naive_crps": None,
        "direction_hit": None,
    }
    if k == 0:
        return out
    last = close[origin - 1]
    b = q[:k]
    lv = np.asarray(levels, dtype=np.float64)
    out["coverage"] = float(((realized >= b[:, lo]) & (realized <= b[:, hi])).mean())
    out["mae"] = float(np.abs(b[:, mid] - realized).mean())
    out["naive_mae"] = float(np.abs(last - realized).mean())
    out["crps"] = float(_crps_steps(b, realized, lv).mean())
    if naive_q is not None:
        out["naive_crps"] = float(_crps_steps(naive_q[:k], realized, lv).mean())
    if k == horizon:
        pred = np.sign(q[horizon - 1, mid] - last)
        real = np.sign(realized[-1] - last)
        out["direction_hit"] = bool(pred != 0 and pred == real)
    return out


class _Aggregate:
    """Running sums over fully realized origins, per step and per quantile level."""

    def __init__(self, horizon: int, levels: Sequence[float]) -> None:
        self.levels = np.asarray(levels, dtype=np.float64)
        self.lo, self.mid, self.hi = _level_indices(levels, (LO, 0.5, HI))
        self.n = 0
        self.hits = 0
        self.inside = np.zeros(horizon)
        self.err = np.zeros(horizon)
        self.naive_err = np.zeros(horizon)
        self.crps = np.zeros(horizon)
        self.naive_crps = np.zeros(horizon)
        self.below = np.zeros(len(levels))
        self.coverage_sum = 0.0

    def add(self, close: np.ndarray, origin: int, q: np.ndarray, naive_q: np.ndarray, scores: dict) -> None:
        h = q.shape[0]
        y = close[origin : origin + h]
        last = close[origin - 1]
        self.n += 1
        self.hits += bool(scores["direction_hit"])
        self.coverage_sum += scores["coverage"]
        self.inside += (y >= q[:, self.lo]) & (y <= q[:, self.hi])
        self.err += np.abs(q[:, self.mid] - y)
        self.naive_err += np.abs(last - y)
        self.crps += _crps_steps(q, y, self.levels)
        self.naive_crps += _crps_steps(naive_q, y, self.levels)
        self.below += (y[:, None] <= q).sum(axis=0)

    def summary(self) -> dict:
        if self.n == 0:
            return {"scored": 0}
        n, h = self.n, len(self.inside)
        mae, naive = self.err.sum() / (n * h), self.naive_err.sum() / (n * h)
        crps, naive_crps = self.crps.sum() / (n * h), self.naive_crps.sum() / (n * h)
        observed = self.below / (n * h)
        return {
            "scored": n,
            "coverage": self.coverage_sum / n,
            "mae": mae,
            "naive_mae": naive,
            # > 0 means the forecast beats the naive one.
            "skill": _skill(mae, naive),
            "crps": crps,
            "naive_crps": naive_crps,
            "crps_skill": _skill(crps, naive_crps),
            "direction_hit_rate": self.hits / n,
            "by_horizon": {
                "coverage": [_round(v, 4) for v in self.inside / n],
                "mae_skill": [_round(_skill(a, b), 4) for a, b in zip(self.err, self.naive_err)],
                "crps_skill": [_round(_skill(a, b), 4) for a, b in zip(self.crps, self.naive_crps)],
            },
            "calibration": {
                "levels": [float(v) for v in self.levels],
                "observed": [_round(v, 4) for v in observed],
            },
            "calibration_error": float(np.abs(observed - self.levels).mean()),
        }


def _skill(model: float, naive: float) -> float | None:
    return float(1.0 - model / naive) if naive > 0 else None


def run_backtest(
    forecaster,
    close: Sequence[float],
    params: BacktestParams,
    progress: Callable[[int, int], None] | None = None,
    volume: Sequence[float] | None = None,
    raw: bool = False,
    past_covariates: np.ndarray | None = None,
    known_covariates: np.ndarray | None = None,
) -> dict:
    """Walk-forward backtest; see the module docstring.

    `past_covariates` `(C, T)` adds rows known only up to each origin (aligned with `close`); `known_covariates`
    `(C, T + horizon)` adds rows known ahead of time, such as the calendar, covering the horizon after the data end.

    With `raw=True` the result also holds `raw`: `index` (N,), `q` (N, H, Q) forecast quantiles and
    `naive_q` (N, H, Q) for every origin, for offline analysis (not JSON-serializable).

    `forecaster` needs `quantile_levels` and `forecast(context (B, 1, T), prediction_length)` returning
    an object with `quantiles` `(B, 1, H, Q)`; with `params.covariates` it also gets
    `past_covariates` `(B, 2, T)` built from `volume`. `progress(done, total)` is called after each chunk.

    Returns `{"levels", "trend", "origins": [{"index", "slope", "bands": (H, 5) list, scores...}],
    "summary"}`. With a trend, `summary["plain"]` scores the untilted forecast.
    """
    series = np.asarray(close, dtype=np.float64)
    n = len(series)
    params.validate(n)
    origins = origin_indices(n, params)
    levels = tuple(float(v) for v in forecaster.quantile_levels)
    band_idx = _level_indices(levels, BAND_LEVELS)
    trend = params.trend_window > 0
    covs = None
    if params.covariates:
        if volume is None or len(volume) != n:
            raise ValueError("covariates need a volume series as long as the prices")
        covs = covariate_rows(series, np.asarray(volume, dtype=np.float64))
    if past_covariates is not None:
        extra = np.asarray(past_covariates, dtype=np.float32)
        if extra.ndim != 2 or extra.shape[1] != n:
            raise ValueError(f"past_covariates must be (C, {n})")
        covs = extra if covs is None else np.concatenate([covs, extra])
    known = None
    if known_covariates is not None:
        known = np.asarray(known_covariates, dtype=np.float32)
        if known.ndim != 2 or known.shape[1] != n + params.horizon:
            raise ValueError(f"known_covariates must be (C, {n + params.horizon})")

    raw_q: list[np.ndarray] = []
    raw_naive: list[np.ndarray] = []
    agg = _Aggregate(params.horizon, levels)
    plain_agg = _Aggregate(params.horizon, levels) if trend else None
    ctx_len = min(params.context, max(origins))
    results: list[dict] = []
    total = len(origins)
    if progress is not None:
        progress(0, total)
    for start in range(0, total, params.batch_size):
        chunk = origins[start : start + params.batch_size]
        batch = np.full((len(chunk), 1, ctx_len), np.nan, dtype=np.float32)
        cov_batch = np.full((len(chunk), len(covs), ctx_len), np.nan, dtype=np.float32) if covs is not None else None
        known_batch = (
            np.full((len(chunk), len(known), ctx_len + params.horizon), np.nan, dtype=np.float32) if known is not None else None
        )
        for row, i in enumerate(chunk):
            lo = max(0, i - ctx_len)
            batch[row, 0, ctx_len - (i - lo) :] = series[lo:i]
            if cov_batch is not None:
                cov_batch[row, :, ctx_len - (i - lo) :] = covs[:, lo:i]
            if known_batch is not None:
                known_batch[row, :, ctx_len - (i - lo) :] = known[:, lo : i + params.horizon]
        extra_kw = {}
        if cov_batch is not None:
            extra_kw["past_covariates"] = cov_batch
        if known_batch is not None:
            extra_kw["known_covariates"] = known_batch
        forecast = forecaster.forecast(batch, prediction_length=params.horizon, **extra_kw)
        q_all = _to_numpy(forecast.quantiles)
        if q_all.shape[:3] != (len(chunk), 1, params.horizon) or q_all.shape[3] != len(levels):
            raise ValueError(f"forecaster returned quantiles shaped {q_all.shape}")
        for row, i in enumerate(chunk):
            plain = q_all[row, 0]
            slope = trend_slope(series, i, params.trend_window) if trend else 0.0
            q = apply_trend(plain, slope) if trend else plain
            naive_q = naive_quantiles(series, i, params.horizon, levels, params.vol_window)
            entry = {"index": i, "slope": _round(slope, 6), "bands": q[:, band_idx]}
            entry.update(score_origin(series, i, q, levels, params.horizon, naive_q))
            if entry["direction_hit"] is not None:
                agg.add(series, i, q, naive_q, entry)
                if plain_agg is not None:
                    plain_agg.add(series, i, plain, naive_q, score_origin(series, i, plain, levels, params.horizon))
            results.append(entry)
            if raw:
                raw_q.append(q)
                raw_naive.append(naive_q)
        if progress is not None:
            progress(len(results), total)

    summary = agg.summary()
    if plain_agg is not None:
        summary["plain"] = plain_agg.summary()
    for entry in results:
        entry["bands"] = [[_round(v) for v in step] for step in entry["bands"]]
    out_raw = None
    if raw:
        out_raw = {
            "index": np.array([e["index"] for e in results]),
            "q": np.stack(raw_q),
            "naive_q": np.stack(raw_naive),
            "levels": np.asarray(levels),
        }
    return {
        "raw": out_raw,
        "levels": list(BAND_LEVELS),
        "trend": trend,
        "covariates": covs is not None or known is not None,
        "origins": results,
        "summary": summary,
    }


def _round(x: float | None, digits: int = 7) -> float | None:
    """Round to `digits` significant digits, keeping JSON small; None and non-finite become None."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")
