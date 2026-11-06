# How well does Timestral forecast crypto prices? A walk-forward study on Binance pairs

coinband project · November 2026 · unofficial, not affiliated with Mistral AI or Binance · not financial advice

## Summary

We backtested Mistral's Timestral time-series forecasters (5M, 28M, 123M and 209M parameters) on hourly
candles of eight Binance spot pairs. We tuned the settings on one period (April to July 2026) and then
scored the chosen settings once on a later period the tuning never saw (July to November 2026). Every
forecast is a full probability band (quantiles), and we score the whole band with CRPS against a random-walk
band built from recent volatility.

- **Timestral's bands beat the random walk, modestly and consistently.** The chosen setup (209M, 4096-hour
  context, price only) scores +1.9% CRPS skill 24 hours ahead in the tuning period and +2.1% in the held-out
  period. All eight pairs are positive in both periods (+0.6% to +4.2%). The bands are well calibrated: the
  80% band holds the price 79 to 82% of the time.
- **The skill is in the band, not the direction.** The median forecast is no better than "price stays flat"
  (MAE skill −0.07%), and the direction of the move 24 hours ahead is called right 51 to 52% of the time.
  Timestral forecasts how far the price may move, not which way.
- **Skill is highest one to six hours ahead** (+2.1 to +3.2% for the default) and fades with the horizon.
- **What helped:** a larger model with a 4096-hour context. Against the previous default (28M, 2048 hours)
  it adds +1.0 points 24 hours ahead in the tuning period (90% CI +0.3 to +1.8) and +0.2 in the held-out
  period (−0.1 to +0.4), and more at 96 hours (+2.7 and +0.6, both intervals above zero).
- **What did not:** adding the recent trend to the forecast (up to −13 points), an 8192-hour context
  (worse than 4096 in both periods, mostly at short horizons), and volume as an extra input beyond a few
  hours ahead (it helps one hour ahead in both periods, and hurts or does nothing 24 hours and more ahead).

## Setup

**Data.** Hourly closed candles from Binance's public market-data API for BTC, ETH, SOL, BNB, XRP, DOGE,
ADA and LINK against USDT, 13,152 hours per pair (May 2025 to November 2026), frozen in a local cache so
every run sees the same data.

**Protocol.** Walk-forward: at each forecast origin the model sees only the past (up to `context` hours,
NaN-padded when shorter) and forecasts the next 96 hours as 21 quantiles. Origins are 8 hours apart. Each
period has 300 origins per pair; the 289 whose 96 hours have fully played out are scored.

- Tuning ("val"): origins from 2026-04-20 to 2026-07-29.
- Held-out ("test"): origins from 2026-07-29 to 2026-11-06.

Screening used the tuning period and four pairs (BTC, ETH, SOL, BNB) for 35 configurations. A shortlist of
eight then ran on all eight pairs in both periods. The default was chosen on the tuning period; held-out
numbers were computed once, after that choice.

**Scores.** All errors are divided by the last known price, so pairs pool fairly regardless of price level.

- *CRPS skill*: 1 − CRPS(forecast) / CRPS(random walk), pooled over pairs and origins, for steps 1 to h.
  CRPS is approximated as twice the mean pinball loss over the model's 21 quantile levels. The random walk
  is the last price times exp(σ·√k·z), with σ the standard deviation of the previous 500 one-hour log returns.
  Above 0 means the forecast band beats it.
- *MAE skill*: the median against "price stays flat".
- *Coverage*: share of realized prices inside the 80% band (P10 to P90).
- *Calibration error*: mean absolute gap between each quantile level and how often the price came in at
  or below it.
- *Direction hit*: whether the median got the sign of the move h hours ahead right (a flat call is a miss).
- *Confidence intervals*: 90% moving-block bootstrap over origins, blocks of 12 origins (96 hours) so
  overlapping forecasts stay together, 1000 resamples, the same resamples for every configuration so
  differences are paired.

## Results

### The shortlist, eight pairs

CRPS skill vs the random walk, % (steps 1 to h):

| Configuration | Tuning h=1 | h=6 | h=24 | h=96 | Held-out h=1 | h=6 | h=24 | h=96 |
|---|---|---|---|---|---|---|---|---|
| **209M, ctx 4096 (default)** | +2.24 | +2.12 | **+1.90** | +0.73 | +3.19 | +2.72 | **+2.13** | +1.51 |
| 209M, ctx 2048 | +3.41 | +2.38 | +1.83 | +0.24 | +3.20 | +2.67 | +2.00 | +1.02 |
| 123M, ctx 4096 | +2.18 | +2.06 | +1.76 | +0.52 | +3.04 | +2.69 | +2.02 | +1.42 |
| 209M, ctx 8192 | −0.42 | +1.37 | +1.65 | +0.38 | +1.26 | +2.18 | +1.93 | +1.07 |
| 28M, ctx 4096 | +2.53 | +1.99 | +1.51 | −0.00 | +2.96 | +2.77 | +2.04 | +1.23 |
| 209M, ctx 4096, volume | +2.98 | +2.55 | +1.25 | −1.01 | +3.85 | +3.42 | +2.20 | +1.57 |
| 28M, ctx 4096, volume | +2.60 | +1.96 | +0.97 | −1.16 | +2.98 | +2.98 | +2.10 | +1.50 |
| 28M, ctx 2048 (old default) | +2.77 | +1.84 | +0.88 | −1.94 | +2.66 | +2.69 | +1.93 | +0.94 |

For the default, 24 hours ahead: MAE skill −0.07% (tuning) and −0.07% (held-out), 80% coverage 82.0% and
79.3%, calibration error 0.88 and 0.80 points, direction hit 50.9% and 51.8%.

![CRPS skill by bars ahead](figures/skill-by-horizon.svg)

The held-out period was easier for every configuration, and the gaps between configurations shrink there.
The tuning period separates them more: there the old default falls below the random walk after about 24
hours, while 209M with 4096 hours stays above it to about 90 hours.

### One change at a time

Change in CRPS skill, percentage points, with 90% paired bootstrap intervals:

| Change | Tuning h=1 | h=24 | h=96 | Held-out h=1 | h=24 | h=96 |
|---|---|---|---|---|---|---|
| Old default to 209M, ctx 4096 | −0.52 (−1.05, +0.09) | **+1.02 (+0.34, +1.78)** | **+2.66 (+0.78, +4.62)** | **+0.53 (+0.11, +0.94)** | +0.20 (−0.07, +0.36) | **+0.57 (+0.14, +0.96)** |
| Context 2048 to 4096, 209M | **−1.16 (−1.50, −0.81)** | +0.07 (−0.19, +0.38) | +0.48 (−0.10, +1.11) | −0.01 (−0.20, +0.17) | +0.13 (−0.01, +0.24) | **+0.49 (+0.11, +0.86)** |
| Context 4096 to 8192, 209M | **−2.66 (−3.35, −2.00)** | **−0.25 (−0.45, −0.08)** | **−0.35 (−0.62, −0.11)** | **−1.93 (−2.50, −1.32)** | −0.20 (−0.40, +0.02) | **−0.43 (−0.76, −0.12)** |
| 28M to 123M, ctx 4096 | −0.35 (−0.82, +0.52) | +0.25 (−0.02, +0.64) | **+0.52 (+0.02, +1.15)** | +0.08 (−0.50, +0.61) | −0.01 (−0.28, +0.26) | +0.18 (−0.49, +0.88) |
| 123M to 209M, ctx 4096 | +0.06 (−0.49, +0.36) | +0.14 (−0.06, +0.32) | +0.21 (−0.15, +0.63) | +0.15 (−0.21, +0.59) | +0.11 (−0.05, +0.26) | +0.09 (−0.12, +0.32) |
| Volume, 209M ctx 4096 | **+0.73 (+0.20, +1.14)** | **−0.65 (−1.36, −0.17)** | **−1.73 (−2.87, −0.97)** | **+0.66 (+0.28, +1.11)** | +0.06 (−0.35, +0.45) | +0.07 (−0.56, +0.65) |

Bold: the interval excludes zero.

- **Model size** helps a little at long horizons. Most single steps up are within noise (only 28M to 123M
  at 96 hours in the tuning period clears zero), and the gain from 28M to 209M is clearer in the tuning period.
- **Context** has a sweet spot. Going from 2048 to 4096 hours helps long horizons but can cost the next
  hour; going to 8192 is worse at every horizon in both periods.
- **Volume** (log volume and the absolute one-hour return as past-only covariates) sharpens the next hour
  in both periods (and the next six in the held-out one), then hurts in the tuning period and does nothing in the held-out one. It also costs
  about six times the inference. We leave it off by default and suggest it for horizons up to six hours.

### Per pair

CRPS skill 24 hours ahead, default configuration:

| | ADA | BNB | BTC | DOGE | ETH | LINK | SOL | XRP |
|---|---|---|---|---|---|---|---|---|
| Tuning | +2.13 | +2.34 | +1.87 | +1.53 | +1.64 | +1.93 | +2.07 | +1.71 |
| Held-out | +1.90 | +0.82 | +2.39 | +2.96 | +2.81 | +0.60 | +1.17 | +4.19 |

The old default was negative on SOL in the tuning period (−0.60) and positive elsewhere.

### What did not work

- **Trend drift.** Tilting every forecast by the least-squares slope of log price over the last N hours
  (screen, four pairs, tuning period, 28M with a 4096-hour context and volume) changed CRPS skill 24 hours
  ahead from +0.95% without trend by −13.3 points (N = 48), −4.6 (N = 168) and −1.1 (N = 720), and was worse
  at every horizon tested. Short-term crypto momentum did not carry forward over this period.
- **Log prices.** In an earlier, smaller test feeding log price instead of price lowered CRPS skill for both
  28M and 209M (by about 0.1 points). Timestral already normalizes its input.
- **Running on more CPU threads or processes.** CPU inference is memory-bandwidth bound on the test machine
  (an M5 Pro): 6, 12 and 18 threads ran at the same speed, and four parallel processes made each run about
  nine times slower. Moving only the transformer to the Apple GPU gave a 3x speedup with outputs equal to
  float32 rounding.

## Limitations

- **Two periods of about 100 days each, one market.** The eight pairs move together, and the bootstrap
  resamples each pair separately, so it ignores the correlation between pairs; the intervals are likely too
  narrow. The ranking of close configurations changed between periods.
- **Hourly crypto only.** Daily data, other intervals and other markets (stocks, indices) are not covered
  here.
- **The baseline matters.** The random walk uses a fixed 500-hour volatility and a normal shape. A stronger
  baseline (GARCH, empirical quantiles of past returns) would shrink the skill numbers.
- **CRPS is approximated** from 21 unevenly spaced quantiles; it is applied the same way to the forecast and
  the baseline.
- **Forecast quality is not trading performance.** There are no costs, no positions and no strategy here.

## Practical defaults

For hourly crypto, use Timestral-209M with a 4096-hour context and price only. Turn volume on for
horizons up to about six hours. Leave the trend off. Expect a band that is well calibrated and a few percent
better than a random walk, and a median that carries no directional edge.

## Reproduce

```bash
uv sync
uv run python experiments/sweep.py --stage screen --device mps     # 35 configs, tuning period, 4 pairs
uv run python experiments/sweep.py --stage confirm --configs experiments/confirm.json --device mps
uv run python experiments/analyze.py --stage confirm --period test --ref test-1h-28M-ctx2048-cov0-trend0-h96
uv run python experiments/effects.py && uv run python experiments/figures.py
```

Data is fetched once into `experiments/cache/`. Results differ slightly if fetched on another day, since the
periods end at the latest closed candle.
