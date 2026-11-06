# coinband

Backtest [Timestral](https://github.com/mistralai/timestral), Mistral AI's open time-series forecaster, on any
Binance coin (or stock, index, FX pair), then watch its forecast band play out against the real price.

> Unofficial. Not affiliated with Mistral AI or Binance. Not financial advice.

- **Read the study:** <https://darkbringer1.github.io/coinband/> (three parts, written for general readers)
- **Try the demo in your browser:** <https://darkbringer1.github.io/coinband/demo/> (no install, precomputed runs)
- **Run it on your machine:** see [Quick start](#quick-start) below

## What you get

- **Walk-forward backtest.** At every forecast origin Timestral sees only the past and forecasts the next
  `horizon` bars as quantiles (P05, P10, P50, P90, P95). Nothing from the future leaks in.
- **Animated playback.** A forecast cone slides along the real price. Red marks show where the price left the band.
  Follow view or full view, scrub with the playhead.
- **Honest scores against simple baselines** (see [Scores](#scores)).
- **Live mode.** Refreshes every 60 s: new closed bars, a fresh forecast past the data end, the current price.
- **Leaderboard.** Same settings over several pairs, ranked. Click a row to replay it.
- **Video export.** "Record .webm" replays the run at 1920x1080 and downloads it.

**Headline result** (from the [study](docs/paper.md)): the bands are well calibrated and beat a random walk by
about 2% CRPS a day ahead on all eight coins tested, but the median has no directional edge.

## Quick start

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/darkbringer1/coinband && cd coinband
uv sync
uv run coinband            # open http://127.0.0.1:8000
```

1. Pick a source (Binance or Yahoo), a symbol and an interval.
2. Press **Run**. The first run of a model downloads its weights from Hugging Face (21 MB for Timestral-5M,
   837 MB for Timestral-209M).
3. Watch the playback, read the scores under the chart, switch on **Live** if you want.

Options: `--host`, `--port`, `--device auto|cpu|cuda|mps` (or `COINBAND_HOST`, `COINBAND_PORT`,
`COINBAND_DEVICE`). On Apple silicon `auto` picks the GPU, about 3x faster for the 209M model.

Shareable links fill the form and start a run:
`http://127.0.0.1:8000/#symbol=ETHUSDT&interval=4h&horizon=12&run`

## Data sources

| Source | Covers | Key needed | Notes |
| --- | --- | --- | --- |
| Binance | every spot pair | no | `data-api.binance.vision` mirror, `api.binance.com` fallback |
| Yahoo Finance | stocks, indices, FX, futures (Nasdaq, BIST, S&P 500, DAX, gold, USD/TRY, ...) | no | unofficial endpoint, terms restrict redistribution |

Bars are cached in `~/.cache/coinband` and updated incrementally. Yahoo data stays in that local cache only: it
is never committed and never published on the website.

## Scores

All scores use only origins whose whole horizon has already played out.

| Score | Meaning | Good value |
| --- | --- | --- |
| 80% coverage | share of real bars inside the P10 to P90 band | close to 80% |
| MAE vs naive | `1 - MAE(median) / MAE(last value)` | above 0 beats "price stays flat" |
| Direction hit | median got the sign of the move at the horizon end | above 50% |
| CRPS skill | whole band vs a random-walk band from recent volatility | above 0 |

Accuracy by bars ahead and a calibration chart sit under the playback, with a table view.

**Optional knobs, off by default.** *Trend window* tilts forecasts by the recent log-price slope; it made scores
worse on every window tried. *Volume covariates* feed log volume and absolute returns as past-only inputs; they
help the next hour or so, cost about 6x the inference. Defaults (Timestral-209M, 4096-bar context, price only)
come from the study.

## The study

Three write-ups in `docs/`, each with a plain-language layer and collapsible expert sections. Every number comes
from a script in `experiments/`.

| Part | Question | Source | Scripts |
| --- | --- | --- | --- |
| 1 | How good are the forecasts, and which settings are best? | [docs/paper.md](docs/paper.md) | `experiments/sweep.py`, `analyze.py`, `effects.py`, `figures.py` |
| 2 | Is it a better risk model than GARCH? VaR, sizing, direction | [docs/risk.md](docs/risk.md) | `experiments/phase1/` |
| 3 | Do extra inputs help? How does it compare to the options market (Deribit DVOL)? | [docs/options.md](docs/options.md) | `experiments/phase2/`, `experiments/phase3/` |

Each document ends with a "Check it yourself" section listing the exact commands. Part 1 takes about half an
hour on an Apple-silicon GPU, several hours on CPU. Caches and results land in `experiments/cache/` and
`experiments/results/` (gitignored).

## Project layout

```
coinband/
  server.py        stdlib http.server: JSON API + serves static/
  backtest.py      walk-forward engine, batched forecasts, per-origin scores
  binance_data.py  Binance symbols and klines (paged backward, open candle dropped)
  yahoo_data.py    Yahoo Finance bars
  markets.py       source registry (Binance, Yahoo)
  cache.py         local bar cache, incremental updates
  sessions.py      trading-session times for bars after the data end
  device.py        runs the transformer on MPS, keeps float64 scaling on CPU
  paper.py         renders docs/*.md (small Markdown subset)
  static/          index.html, app.js, style.css: vanilla JS + <canvas>, no build step
experiments/       study scripts and the site builder
docs/              study sources (.md) and the generated GitHub Pages site
tests/             engine and API tests with a fake forecaster (no weights needed)
```

### API

| Method | Path | Does |
| --- | --- | --- |
| GET | `/api/markets` | data sources, their intervals and defaults |
| GET | `/api/symbols?source=` | symbols for a source |
| GET | `/api/models` | Timestral checkpoints and which are loaded |
| POST | `/api/backtest` | start a backtest job, returns `{"job_id"}` |
| POST | `/api/leaderboard` | same settings over several symbols, returns `{"job_id"}` |
| GET | `/api/jobs` | the queue: running, waiting, recent |
| GET | `/api/jobs/<id>` | status, progress and, once done, the result |
| POST | `/api/jobs/<id>/cancel` | stop a job |
| POST | `/api/live` | new closed bars, a fresh live forecast and the current price |

## Development

```bash
uv sync
uv run pytest              # fast, no model weights or network needed
```

Ground rules (the full list is in [CLAUDE.md](CLAUDE.md)):

- **No new runtime dependencies.** Only `timestral`, pinned to upstream commit `a534bdb`. Backend is stdlib,
  frontend is vanilla JS with no build step and no CDN.
- **Do not vendor or copy timestral code.** Wrap it at runtime (see `device.py`). Upstream does not take PRs.
- **Never commit or publish Yahoo data.** Only Binance data may go into `docs/`.
- **Numbers in the docs must come from the scripts' outputs.** Change a script, rerun it, then update the text.
- **Never present this as official Mistral or Binance work.** No Mistral logos.

## Website (GitHub Pages)

`docs/` is the site, served from the `main` branch, `/docs` folder:

- `docs/index.html`, `docs/risk/`, `docs/options/`: the three parts, rendered from the `.md` files
- `docs/demo/`: a copy of `coinband/static` in static mode, replaying precomputed Timestral-209M runs for eight
  Binance coins from `docs/demo/data/`

The HTML is generated. Do not edit it by hand; edit the `.md` files or `coinband/static/` and rebuild:

```bash
uv run python experiments/build_site.py --pages-only     # re-render pages and demo UI, keep the data
uv run python experiments/build_site.py --device mps     # also refresh the demo forecasts (a few minutes on a GPU)
```

Preview locally with `python -m http.server -d docs 8001`, then open <http://127.0.0.1:8001/>. Push to `main`
to publish.

## For AI agents

Start with [CLAUDE.md](CLAUDE.md): it lists decisions already made and the target layout. Then:

1. Run `uv run pytest` before and after a change. Tests use a fake forecaster, so they are fast and offline.
2. Keep changes inside the existing modules; do not add packages to `pyproject.toml`.
3. After touching `docs/*.md` or `coinband/static/`, run `experiments/build_site.py --pages-only` and commit the
   regenerated `docs/` files with the change.
4. Do not touch `docs/demo/data/` unless asked to refresh the demo; it needs model weights and a few minutes of
   GPU time.

## License

Code: [Apache 2.0](LICENSE). Timestral weights download from the Hugging Face Hub under their own
[Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE).
