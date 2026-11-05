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

## License

Code: [Apache 2.0](LICENSE). Timestral weights are downloaded from the Hugging Face Hub and are under their own
[Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE).
