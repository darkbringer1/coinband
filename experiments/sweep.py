"""Accuracy sweep for the coinband write-up: which Timestral settings forecast Binance pairs best.

Every run is one config (model, context, covariates, trend, interval) on a fixed set of pairs and a fixed
set of origins, so configs can be compared origin by origin. Origins are `stride` bars apart; with a
stride below the horizon neighbouring forecasts overlap, which the bootstrap in `analyze.py` accounts for.

Two periods, split in time:

- `val`: the `n_origins * stride` bars before the test period. Use it to pick settings.
- `test`: the last `n_origins * stride` bars. Report it, once, for the settings picked on `val`.

Per pair the script saves normalized per-origin, per-step arrays to `results/<stage>/<run_id>/<SYMBOL>.npz`:
CRPS and absolute error of the forecast and of the naive baselines (all divided by the last known price,
so pairs pool fairly), inside-the-80%-band flags, calibration hits per quantile level and the direction
call at every step. `analyze.py` turns those into tables with confidence intervals.

    uv run python experiments/sweep.py --stage screen     # val period, 4 pairs, the full grid
    uv run python experiments/sweep.py --stage confirm    # val + test, 8 pairs, the shortlist

Finished runs are skipped, so a stopped sweep resumes where it left off. `--workers N` downloads all data
first, then runs N configs at a time in separate processes, longest first, splitting the CPU threads.
On an M5 Pro CPU inference is memory-bandwidth bound (6 to 18 threads gave the same speed), and 4 workers ran
each config about 9x slower, so the default of 1 is the fastest there; try more only on a GPU or a big server.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from coinband.backtest import BacktestParams, _crps_steps, run_backtest
from coinband.binance_data import fetch_klines

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"

MODELS = ["5M", "28M", "123M", "209M"]
SCREEN_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
CONFIRM_PAIRS = SCREEN_PAIRS + ["XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]
HORIZON = 96
N_ORIGINS = 300
STRIDE = 8


def load_klines(symbol: str, interval: str, bars: int):
    """Klines from the local cache, fetched once; the cache pins the data so reruns are comparable."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{symbol}-{interval}-{bars}.json"
    if path.exists():
        d = json.loads(path.read_text())
    else:
        k = fetch_klines(symbol, interval, bars)
        d = {"time": k.open_time, "close": k.close, "volume": k.volume}
        path.write_text(json.dumps(d))
    return np.asarray(d["time"]), np.asarray(d["close"], dtype=np.float64), np.asarray(d["volume"], dtype=np.float64)


_models: dict[str, object] = {}
DEVICE = "cpu"


def get_model(size: str):
    if size not in _models:
        from coinband.device import load_forecaster

        _models[size] = load_forecaster(f"mistralai/Timestral-{size}", DEVICE)
    return _models[size]


def run_id(cfg: dict) -> str:
    return (
        f"{cfg['period']}-{cfg['interval']}-{cfg['model']}-ctx{cfg['context']}"
        f"-cov{int(cfg['covariates'])}-trend{cfg['trend_window']}-h{cfg['horizon']}"
    )


def evaluate(close: np.ndarray, raw: dict, horizon: int) -> dict[str, np.ndarray]:
    """Per-origin, per-step arrays over fully realized origins, normalized by the last known price."""
    idx = raw["index"]
    keep = idx + horizon <= len(close)
    idx, q, nq = idx[keep], raw["q"][keep], raw["naive_q"][keep]
    levels = raw["levels"]
    lo, mid, hi = (int(np.argmin(np.abs(levels - v))) for v in (0.1, 0.5, 0.9))
    y = np.stack([close[i : i + horizon] for i in idx])  # (N, H)
    last = close[idx - 1][:, None]  # (N, 1)
    crps = np.stack([_crps_steps(q[n], y[n], levels) for n in range(len(idx))]) / last
    naive_crps = np.stack([_crps_steps(nq[n], y[n], levels) for n in range(len(idx))]) / last
    return {
        "index": idx,
        "crps": crps.astype(np.float32),
        "naive_crps": naive_crps.astype(np.float32),
        "ae": (np.abs(q[:, :, mid] - y) / last).astype(np.float32),
        "naive_ae": (np.abs(last - y) / last).astype(np.float32),
        "inside": ((y >= q[:, :, lo]) & (y <= q[:, :, hi])),
        "below": (y[:, :, None] <= q),  # (N, H, Q) calibration hits
        "levels": levels,
        # direction call of the median vs the realized move, at every step
        "dir_pred": np.sign(q[:, :, mid] - last).astype(np.int8),
        "dir_real": np.sign(y - last).astype(np.int8),
        "width": ((q[:, :, hi] - q[:, :, lo]) / last).astype(np.float32),
    }


def run_config(cfg: dict, pairs: list[str], stage: str) -> None:
    out = RESULTS / stage / run_id(cfg)
    done = out / "done.json"
    if done.exists():
        print(f"skip {out.name}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    span = N_ORIGINS * STRIDE
    bars = cfg["fetch_bars"]
    model = get_model(cfg["model"])
    started = time.time()
    summary = {}
    for symbol in pairs:
        if (out / f"{symbol}.npz").exists():
            continue
        _, close, volume = load_klines(symbol, cfg["interval"], bars)
        # val ends where test begins; each period scores its own origins on data up to its end
        end = len(close) if cfg["period"] == "test" else len(close) - span
        c, v = close[:end], volume[:end]
        params = BacktestParams(
            context=cfg["context"],
            horizon=cfg["horizon"],
            stride=STRIDE,
            max_origins=N_ORIGINS + 1,  # +1: the newest origin has nothing realized and is dropped
            batch_size=32,
            covariates=cfg["covariates"],
            trend_window=cfg["trend_window"],
        )
        bt = run_backtest(model, c, params, volume=v, raw=True)
        arrays = evaluate(c, bt["raw"], cfg["horizon"])
        tmp = out / f".{symbol}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out / f"{symbol}.npz")  # atomic: a killed worker never leaves a half-written file
        skill = 1 - arrays["crps"].sum() / arrays["naive_crps"].sum()
        summary[symbol] = round(float(skill) * 100, 2)
        print(f"  {out.name} {symbol}: CRPS skill {summary[symbol]:+.2f}% ({len(arrays['index'])} origins)", flush=True)
    done.write_text(json.dumps({**cfg, "pairs": pairs, "seconds": round(time.time() - started, 1)}))
    print(f"done {out.name} in {time.time() - started:.0f}s", flush=True)


COST = {"5M": 1, "28M": 2, "123M": 6, "209M": 8}


def cost(cfg: dict) -> float:
    return COST[cfg["model"]] * cfg["context"] * (6 if cfg["covariates"] else 1)


def _init_worker(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def _run_job(job: tuple[dict, list[str], str]) -> str:
    cfg, pairs, stage = job
    run_config(cfg, pairs, stage)
    return run_id(cfg)


def run_jobs(jobs: list[tuple[dict, list[str], str]], workers: int) -> None:
    # download everything up front so workers never fetch (or write the cache) concurrently
    for cfg, pairs, _ in jobs:
        for symbol in pairs:
            load_klines(symbol, cfg["interval"], cfg["fetch_bars"])
    if workers <= 1:
        for job in jobs:
            _run_job(job)
        return
    jobs = sorted(jobs, key=lambda j: -cost(j[0]))  # longest first packs the pool best
    threads = max(1, (os.cpu_count() or workers) // workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(threads,), maxtasksperchild=1) as pool:
        for name in pool.imap_unordered(_run_job, jobs):
            print(f"worker finished {name}", flush=True)


def screen_configs() -> list[dict]:
    """The full grid on the val period, cheapest first so results stream in early."""
    grid = []
    for model, context, cov in itertools.product(MODELS, [1024, 2048, 4096, 8192], [False, True]):
        grid.append(
            {"period": "val", "interval": "1h", "model": model, "context": context, "covariates": cov,
             "trend_window": 0, "horizon": HORIZON, "fetch_bars": 8192 + 2 * N_ORIGINS * STRIDE + HORIZON + 64}
        )
    # trend on the strongest cheap config, to confirm it does not help with more data
    for tw in (48, 168, 720):
        grid.append({**grid[0], "model": "28M", "context": 4096, "covariates": True, "trend_window": tw})
    return sorted(grid, key=cost)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "confirm"], required=True)
    ap.add_argument("--configs", help="confirm stage: JSON file with a list of configs to run on val and test")
    ap.add_argument("--workers", type=int, default=1, help="configs run in parallel (default 1)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or mps (mps: transformer on the GPU)")
    args = ap.parse_args(argv)
    global DEVICE
    DEVICE = args.device
    if args.stage == "screen":
        jobs = [(cfg, SCREEN_PAIRS, "screen") for cfg in screen_configs()]
    else:
        if not args.configs:
            sys.exit("--configs is required for the confirm stage")
        jobs = [
            ({**cfg, "period": period}, CONFIRM_PAIRS, "confirm")
            for cfg in json.loads(Path(args.configs).read_text())
            for period in ("val", "test")
        ]
    run_jobs(jobs, args.workers)


if __name__ == "__main__":
    main()
