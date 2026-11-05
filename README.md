# coinband

Backtest [Timestral](https://github.com/mistralai/timestral), Mistral AI's time-series foundation model, on any
Binance coin, and watch the forecast band play out against the real price.

> Work in progress. Not affiliated with Mistral AI or Binance. Not financial advice.

## What it does

- Pulls candles for any Binance spot pair (public market-data API, no key).
- Runs a walk-forward backtest: at each forecast origin, Timestral sees only the past and forecasts the next
  `horizon` bars as quantiles.
- Animates the run in a browser: the real price is revealed bar by bar, with the 80% band (P10 to P90) and the
  median overlaid, and live scores against a naive "price stays flat" baseline.

## Run

```bash
uv sync
uv run coinband          # http://127.0.0.1:8000  (--port, --host, --device cpu|cuda|mps)
uv run pytest
```

The first run of each model downloads its weights from the Hugging Face Hub (21 MB for Timestral-5M, 837 MB for
Timestral-209M). Links like `http://127.0.0.1:8000/#symbol=ETHUSDT&interval=4h&horizon=12&run` fill the form and
start the run. **Record .webm** replays the run at 1920x1080 and downloads the video.

Scores, all over origins whose whole horizon has played out:

- **80% coverage**: share of realized bars inside the P10 to P90 band. A calibrated model sits near 80%.
- **MAE vs naive**: `1 - MAE(median) / MAE(last value)`. Above 0 means the median beats "price stays flat".
- **Direction hit**: whether the median got the sign of the move at the horizon end right.

## License

Code: [Apache 2.0](LICENSE). Timestral weights are downloaded from the Hugging Face Hub and are under their own
[Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE).
