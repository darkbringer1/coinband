import math
from types import SimpleNamespace

import numpy as np
import pytest

from coinband.backtest import (
    BAND_LEVELS,
    BacktestParams,
    apply_trend,
    naive_quantiles,
    origin_indices,
    run_backtest,
    score_origin,
    trend_slope,
)

LEVELS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


class FakeForecaster:
    """Forecasts the last observed value with a fixed spread per quantile level; records every call."""

    quantile_levels = LEVELS

    def __init__(self, spread=1.0, drift=0.0):
        self.spread = spread
        self.drift = drift
        self.calls = []
        self.covariates = []

    def forecast(self, context, prediction_length, past_covariates=None):
        context = np.asarray(context)
        self.calls.append(context.copy())
        self.covariates.append(None if past_covariates is None else np.asarray(past_covariates).copy())
        b = context.shape[0]
        last = np.array([row[0][np.isfinite(row[0])][-1] for row in context])
        offsets = np.array([(q - 0.5) * 2 * self.spread for q in LEVELS])
        steps = np.arange(1, prediction_length + 1) * self.drift
        q = last[:, None, None, None] + steps[None, None, :, None] + offsets[None, None, None, :]
        return SimpleNamespace(quantiles=q)


def test_origins_end_on_live_and_respect_stride():
    params = BacktestParams(context=100, horizon=5, stride=7, max_origins=1000, min_context=10)
    origins = origin_indices(50, params)
    assert origins[-1] == 50
    assert all(b - a == 7 for a, b in zip(origins, origins[1:]))
    assert origins[0] >= 10 and origins[0] - 7 < 10


def test_max_origins_keeps_the_most_recent():
    params = BacktestParams(horizon=5, stride=1, max_origins=3, min_context=10)
    assert origin_indices(100, params) == [98, 99, 100]


def test_run_backtest_shapes_chunking_and_padding():
    close = np.linspace(100, 200, 300)
    fc = FakeForecaster()
    params = BacktestParams(context=128, horizon=12, stride=12, max_origins=1000, min_context=32, batch_size=5)
    out = run_backtest(fc, close, params)

    origins = out["origins"]
    expected = origin_indices(300, params)
    assert [o["index"] for o in origins] == expected
    assert len(fc.calls) == math.ceil(len(expected) / 5)
    assert out["levels"] == list(BAND_LEVELS)
    for o in origins:
        assert len(o["bands"]) == 12 and all(len(step) == 5 for step in o["bands"])

    # Every context is the same width; short ones are NaN-padded on the left, never on the right.
    first_batch = fc.calls[0]
    assert first_batch.shape[1:] == (1, 128)
    first_origin = expected[0]
    row = first_batch[0, 0]
    assert np.isnan(row[: 128 - first_origin]).all()
    np.testing.assert_allclose(row[128 - first_origin :], close[:first_origin], rtol=1e-6)
    # A long-enough context holds exactly the `context` bars before the origin.
    later = next(i for i in expected if i >= 128)
    k = expected.index(later)
    np.testing.assert_allclose(fc.calls[k // 5][k % 5, 0], close[later - 128 : later], rtol=1e-6)


def test_live_origin_has_no_scores():
    close = np.arange(1.0, 201.0)
    out = run_backtest(FakeForecaster(), close, BacktestParams(context=64, horizon=8, stride=8, min_context=16))
    live = out["origins"][-1]
    assert live["index"] == 200
    assert live["realized"] == 0
    assert live["coverage"] is None and live["direction_hit"] is None


def test_bands_are_quantile_order_and_follow_last_value():
    close = np.full(120, 50.0)
    out = run_backtest(FakeForecaster(spread=2.0), close, BacktestParams(context=64, horizon=4, stride=4, min_context=16))
    for o in out["origins"]:
        for step in o["bands"]:
            assert step == sorted(step)
            assert step[2] == pytest.approx(50.0)
            assert step[1] == pytest.approx(50.0 - 0.8 * 2.0)  # P10
            assert step[3] == pytest.approx(50.0 + 0.8 * 2.0)  # P90


BL = (0.05, 0.1, 0.5, 0.9, 0.95)


def test_score_origin_coverage_mae_and_direction():
    close = np.array([10.0, 10.0, 10.0, 11.0, 13.0, 12.0])
    # origin 3: last known = 10, realized = [11, 13, 12]
    bands = np.array(
        [
            # P05, P10, P50, P90, P95
            [8.0, 9.0, 11.0, 12.0, 13.0],  # 11 inside
            [8.0, 9.0, 11.0, 12.0, 13.0],  # 13 outside
            [8.0, 9.0, 11.0, 12.0, 13.0],  # 12 inside (edge)
        ]
    )
    s = score_origin(close, 3, bands, BL, 3)
    assert s["realized"] == 3
    assert s["coverage"] == pytest.approx(2 / 3)
    assert s["mae"] == pytest.approx((0 + 2 + 1) / 3)
    assert s["naive_mae"] == pytest.approx((1 + 3 + 2) / 3)
    assert s["direction_hit"] is True
    assert s["crps"] > 0 and s["naive_crps"] is None

    down = bands.copy()
    down[:, 2] = 9.0
    assert score_origin(close, 3, down, BL, 3)["direction_hit"] is False


def test_crps_is_zero_for_a_perfect_point_forecast():
    close = np.array([10.0, 10.0, 11.0, 12.0])
    q = np.repeat(close[2:, None], 5, axis=1)
    assert score_origin(close, 2, q, BL, 2)["crps"] == pytest.approx(0.0)


def test_partial_horizon_scores_without_direction():
    close = np.array([10.0, 10.0, 11.0])
    bands = np.tile([8.0, 9.0, 10.0, 12.0, 13.0], (4, 1))
    s = score_origin(close, 2, bands, BL, 4)
    assert s["realized"] == 1
    assert s["coverage"] == 1.0
    assert s["direction_hit"] is None


def test_summary_skill_against_naive():
    close = np.exp(np.arange(400) * 0.01)  # steady uptrend: a drifting forecaster beats "flat"
    out = run_backtest(
        FakeForecaster(spread=0.0, drift=0.0), close, BacktestParams(context=64, horizon=10, stride=10, min_context=16)
    )
    s = out["summary"]
    assert s["scored"] == len(out["origins"]) - 1
    assert s["skill"] == pytest.approx(0.0, abs=1e-5)  # the fake median is "flat"
    assert set(s["by_horizon"]) == {"coverage", "mae_skill", "crps_skill"}
    assert len(s["by_horizon"]["coverage"]) == 10
    assert s["calibration"]["levels"] == list(LEVELS)


def test_trend_drift_fits_the_slope_and_beats_plain_on_a_trend():
    close = np.exp(np.arange(400) * 0.01)
    assert trend_slope(close, 200, 50) == pytest.approx(0.01)
    tilted = apply_trend(np.ones((3, 2)), 0.01)
    np.testing.assert_allclose(tilted[:, 0], np.exp([0.01, 0.02, 0.03]))

    out = run_backtest(
        FakeForecaster(spread=0.0), close, BacktestParams(context=64, horizon=10, stride=10, min_context=16, trend_window=50)
    )
    s = out["summary"]
    assert out["trend"] is True
    assert s["skill"] == pytest.approx(1.0, abs=1e-4)
    assert s["direction_hit_rate"] == 1.0
    assert s["plain"]["skill"] == pytest.approx(0.0, abs=1e-5)
    assert out["origins"][5]["slope"] == pytest.approx(0.01)


def test_calibration_of_a_matching_random_walk():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    naive = naive_quantiles(close, 1500, 4, LEVELS, 500)
    assert naive.shape == (4, len(LEVELS))
    assert np.all(np.diff(naive, axis=1) > 0)


def test_rejects_missing_quantile_levels():
    fc = FakeForecaster()
    fc.quantile_levels = (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="0.05"):
        run_backtest(fc, np.ones(100), BacktestParams(context=32, horizon=4, min_context=16))


def test_too_few_bars():
    with pytest.raises(ValueError, match="at least"):
        run_backtest(FakeForecaster(), np.ones(10), BacktestParams(min_context=64))


def test_progress_reports_every_chunk():
    seen = []
    run_backtest(
        FakeForecaster(),
        np.ones(200),
        BacktestParams(context=32, horizon=4, stride=4, min_context=16, batch_size=10),
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen[0][0] == 0 and seen[-1][0] == seen[-1][1]
    assert all(b[0] > a[0] for a, b in zip(seen, seen[1:]))


def test_covariates_are_aligned_with_the_context():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    volume = rng.uniform(1, 1000, 300)
    fc = FakeForecaster()
    params = BacktestParams(context=64, horizon=4, stride=4, min_context=16, batch_size=8, covariates=True)
    out = run_backtest(fc, close, params, volume=volume)
    assert out["covariates"] is True
    origins = origin_indices(300, params)
    cov = fc.covariates[-1]
    i = origins[-1]
    assert cov.shape == (len(origins) % 8 or 8, 2, 64)
    np.testing.assert_allclose(cov[-1, 0], np.log1p(volume[i - 64 : i]), rtol=1e-5)
    np.testing.assert_allclose(cov[-1, 1], np.abs(np.diff(np.log(close[i - 65 : i]))), rtol=1e-4)
    # an early origin is NaN-padded on the left, like its price context
    first = fc.covariates[0][0]
    assert np.isnan(first[:, : 64 - origins[0]]).all()


def test_covariates_need_volume():
    with pytest.raises(ValueError, match="volume"):
        run_backtest(FakeForecaster(), np.ones(100), BacktestParams(context=32, horizon=4, min_context=16, covariates=True))
