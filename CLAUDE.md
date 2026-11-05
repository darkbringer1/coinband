# coinband

Web UI for backtesting Mistral's Timestral forecaster on Binance coins, with an animated playback that overlays the
forecast band on the real price. Planned to be open-sourced along with a blog post.

## Decisions already made

- Standalone repo. `timestral` comes in as a git dependency pinned to upstream commit `a534bdb`. Do not vendor its
  code. Mistral does not accept PRs upstream.
- No extra runtime deps beyond `timestral`: stdlib `http.server` backend, vanilla JS + `<canvas>` frontend, no
  build step, no CDN.
- Binance data: `https://data-api.binance.vision` (market-data-only mirror, no key, works where `api.binance.com`
  is geo-blocked), with `api.binance.com` as fallback. Klines are paged backward 1000 at a time. Drop the
  still-open last candle.
- GitHub repo `darkbringer1/coinband` stays private until the blog post is ready.
- Never present it as official Mistral or Binance work. No Mistral logos.

## Layout (target)

- `coinband/binance_data.py`: symbol list (exchangeInfo, cached) and kline fetching.
- `coinband/backtest.py`: walk-forward engine. Batches every origin into chunked `forecaster.forecast` calls, using
  NaN left-pad for short contexts. Returns P05/P10/P50/P90/P95 per origin plus per-origin scores: 80% coverage,
  median abs error vs naive last-value, direction hit at horizon end. The last origin is the live forecast past the
  data end.
- `coinband/server.py`: JSON API (`/api/symbols`, `/api/models`, `POST /api/backtest` returns a job id,
  `GET /api/jobs/<id>` reports progress and the result) and serves `static/`. Model instances are cached and a
  lock guards inference.
- `coinband/static/`: `index.html`, `app.js`, `style.css`. Playhead animation with a follow or full view, a trail
  of past bands, the current fan ahead of the playhead, red marks where the price leaves the band, live metrics,
  and canvas `MediaRecorder` export to `.webm` for the blog.
- `tests/`: engine tests with a fake forecaster (no weights needed).

## Run

```bash
uv sync
uv run coinband        # http://127.0.0.1:8000
uv run pytest
```
