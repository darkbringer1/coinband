# coinband

Backtest [Timestral](https://github.com/mistralai/timestral), Mistral AI's time-series foundation model, on any
Binance coin, and watch the forecast band play out against the real price.

> Work in progress. Not affiliated with Mistral AI or Binance. Not financial advice.

## What it does

- Pulls candles for any Binance spot pair (public market-data API, no key).
- Runs a walk-forward backtest: at each forecast origin, Timestral sees only the past and forecasts the next
  `horizon` bars as quantiles.
- Animates the run in a browser: a forecast cone (P05 to P95, median dashed) slides along the full real price
  line, so you see each forecast against what actually happened, with live scores against naive baselines.

## Run

```bash
uv sync
uv run coinband          # http://127.0.0.1:8000  (--port, --host, --device auto|cpu|cuda|mps)
uv run pytest
```

The first run of each model downloads its weights from the Hugging Face Hub (21 MB for Timestral-5M, 837 MB for
Timestral-209M). Links like `http://127.0.0.1:8000/#symbol=ETHUSDT&interval=4h&horizon=12&run` fill the form and
start the run. **Record .webm** replays the run at 1920x1080 and downloads the video.

Scores, all over origins whose whole horizon has played out:

- **80% coverage**: share of realized bars inside the P10 to P90 band. A calibrated model sits near 80%.
- **MAE vs naive**: `1 - MAE(median) / MAE(last value)`. Above 0 means the median beats "price stays flat".
- **Direction hit**: whether the median got the sign of the move at the horizon end right.
- **CRPS skill**: scores the whole band, not just the median (CRPS approximated by twice the mean pinball loss
  over the model's quantile levels), against a naive random-walk band built from recent volatility.
- **Accuracy by bars ahead** and **calibration** (how often the price came in at or below each quantile) are
  charted under the playback, with a table view.

**Trend window** (off by default) tilts every forecast by the recent log-price slope. It is a separate model,
Timestral plus a momentum drift, and the plain forecast is scored next to it so you can see whether it helps.
On 1h BTC, ETH and SOL it made the scores worse at every window tried.

**Volume covariates** (off by default) feed log volume and the absolute 1-bar return to the model as past-only
covariates. They sharpen the next hour or so but hurt or do nothing further out, at about 6x the inference cost.
On Apple silicon the transformer runs on the GPU (`mps`, picked automatically) while timestral's float64 input
scaling stays on the CPU (`coinband/device.py`); that is about 3x faster for 209M and matches the CPU to float32
rounding.

**Defaults** (Timestral-209M, 4096-bar context, price only) come from a walk-forward study on eight Binance
pairs with a held-out period: see [docs/paper.md](docs/paper.md). In short, the bands beat a random walk by about
2% CRPS 24 hours ahead and are well calibrated, but the median has no directional edge. The scripts are in
`experiments/`.

The **leaderboard** runs the same settings on several pairs and ranks them; click a row to replay it.

## License

Code: [Apache 2.0](LICENSE). Timestral weights are downloaded from the Hugging Face Hub and are under their own
[Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE).
