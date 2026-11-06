"""Phase 1, step 0: hourly Timestral forecasts for the risk study.

For each of the eight pairs, every hour from the first one with a full context to the end of the cached data,
Timestral-209M (4096-hour context, price only) forecasts the next 24 hours as 21 quantiles. Saved, unscored,
to `experiments/results/phase1/<PAIR>.npz` (`index`, `q` as log returns vs the last price, `levels`).

    uv run python experiments/phase1/collect.py --device mps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sweep import CONFIRM_PAIRS, load_klines  # noqa: E402

from coinband.backtest import BacktestParams, run_backtest  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "phase1"
CONTEXT, HORIZON, BARS = 4096, 24, 13152
MODEL = "mistralai/Timestral-209M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    from coinband.device import load_forecaster

    OUT.mkdir(parents=True, exist_ok=True)
    model = load_forecaster(MODEL, args.device)
    for symbol in CONFIRM_PAIRS:
        path = OUT / f"{symbol}.npz"
        if path.exists():
            print(f"skip {symbol}", flush=True)
            continue
        _, close, volume = load_klines(symbol, "1h", BARS)
        n_origins = len(close) - CONTEXT + 1
        params = BacktestParams(context=CONTEXT, horizon=HORIZON, stride=1, max_origins=n_origins, batch_size=64)
        started = time.time()
        bt = run_backtest(model, close, params, volume=volume, raw=True)
        raw = bt["raw"]
        last = close[raw["index"] - 1][:, None, None]
        q = np.log(np.maximum(raw["q"], 1e-12) / last).astype(np.float32)
        tmp = OUT / f".{symbol}.tmp.npz"
        np.savez_compressed(tmp, index=raw["index"], q=q, levels=raw["levels"])
        tmp.replace(path)
        print(f"{symbol}: {len(raw['index'])} origins in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
