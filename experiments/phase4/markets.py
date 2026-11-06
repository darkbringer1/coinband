"""Phase 4: does Timestral beat the random walk outside crypto? Stocks, indices, FX and gold from Yahoo.

Same scoring as the first study (`sweep.py`): CRPS, MAE, 80% coverage, calibration and direction against the
naive random walk, on a tuning (`val`) period and a later held-out (`test`) period, with a moving-block bootstrap.
The settings are the crypto defaults or the nearest that fit Yahoo's history, fixed before looking at these
markets, so both periods are out of sample for this question.

Groups (bars, not hours: stock bars only exist in trading hours, so one stock "day" is about 7 US or 9 BIST bars):

- `fx-1h`: six FX pairs and gold futures, hourly. Yahoo keeps 729 days (about 12k bars), enough for the crypto
  default exactly: 209M, 4096-bar context, 96-bar horizon, origins every 8 bars, 300 per period.
- `stocks-1h`: US indices and large caps plus BIST, hourly. Only about 3.5k bars, so a 1024-bar context, a
  35-bar horizon (about a US trading week), origins every 4 bars, 250 per period.
- `daily`: every symbol above on daily bars (20+ years): 2048-bar context, 20-bar horizon (about a month),
  origins every 5 bars, 300 per period.

Daily FX bars (not USD/TRY) are cleaned of one-day spikes that revert the next day: Yahoo has nine such bad
prints in 2008-2009 for EUR/USD, USD/JPY and USD/CHF (see `clean_spikes`).

Yahoo's terms restrict redistribution, so the bars are pinned in a snapshot under `~/.cache/coinband/experiments`
(or `$COINBAND_CACHE`), never in the repository; only scores go to `experiments/results/phase4` (gitignored).
Delete the snapshot to refetch newer data.

    uv run python experiments/phase4/markets.py collect --device mps
    uv run python experiments/phase4/markets.py evaluate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze import block_indices, boot_skill, metrics  # noqa: E402
from sweep import evaluate  # noqa: E402

from coinband import yahoo_data  # noqa: E402
from coinband.backtest import BacktestParams, run_backtest  # noqa: E402
from coinband.cache import cache_dir  # noqa: E402

OUT = HERE.parent / "results" / "phase4"
N_BOOT = 1000

FX = ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CHF=X", "TRY=X", "GC=F"]
STOCKS = ["^GSPC", "^NDX", "AAPL", "MSFT", "NVDA", "AMZN", "XU100.IS", "THYAO.IS", "GARAN.IS"]

# per group: symbols, interval, origin spacing and count per period, scored horizons, and the configs to run
# (the first config is the main one; the others vary one setting)
GROUPS: dict[str, dict] = {
    "fx-1h": {
        "symbols": FX, "interval": "1h", "stride": 8, "n_origins": 300, "horizon": 96, "horizons": (1, 6, 24, 96),
        "configs": [{"model": "209M", "context": 4096}, {"model": "209M", "context": 2048}],
    },
    "stocks-1h": {
        "symbols": STOCKS, "interval": "1h", "stride": 4, "n_origins": 250, "horizon": 35, "horizons": (1, 7, 35),
        "configs": [{"model": "209M", "context": 1024}, {"model": "209M", "context": 512}],
    },
    "daily": {
        "symbols": STOCKS + FX, "interval": "1d", "stride": 5, "n_origins": 300, "horizon": 20, "horizons": (1, 5, 20),
        "configs": [{"model": "209M", "context": 2048}, {"model": "209M", "context": 512}],
    },
}

_models: dict[str, object] = {}


def get_model(size: str, device: str):
    if size not in _models:
        from coinband.device import load_forecaster

        _models[size] = load_forecaster(f"mistralai/Timestral-{size}", device)
    return _models[size]


def snapshot_path(symbol: str, interval: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
    return cache_dir() / "experiments" / "yahoo" / f"{safe}-{interval}.json"


def clean_spikes(close: np.ndarray, k: float = 8.0) -> tuple[np.ndarray, int]:
    """Replace one-bar spikes that revert the next bar (bad prints) with the previous close.

    A spike is a log return above `k` robust standard deviations (MAD) followed by a return that undoes at least
    75% of it. Only used for daily FX: Yahoo's daily FX history has such prints (EUR/USD +16% and back on
    2008-12-08), while stocks and indices have real one-day reversals of that size (S&P 500, March 2020).
    """
    c = close.copy()
    r = np.diff(np.log(c))
    sigma = np.median(np.abs(r)) * 1.4826
    n = 0
    for t in range(len(r) - 1):
        if abs(r[t]) > k * sigma and abs(r[t] + r[t + 1]) < 0.25 * abs(r[t]):
            c[t + 1] = c[t]
            n += 1
    return c, n


def load_bars(symbol: str, interval: str) -> np.ndarray:
    """Closes from the pinned snapshot, fetched once, so reruns score the same bars; daily FX is despiked."""
    path = snapshot_path(symbol, interval)
    if not path.exists():
        k = yahoo_data.fetch_klines(symbol, interval, 1_000_000)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"time": k.open_time, "close": k.close, "fetched_at": int(time.time())}))
        os.replace(tmp, path)
    close = np.asarray(json.loads(path.read_text())["close"], dtype=np.float64)
    # not USD/TRY: its big one-day reversals are real (the 2014 emergency rate hike, the December 2021 crisis)
    if interval == "1d" and symbol.endswith("=X") and symbol != "TRY=X":
        close, _ = clean_spikes(close)
    return close


def run_id(group: str, period: str, cfg: dict) -> str:
    return f"{period}-{GROUPS[group]['interval']}-{cfg['model']}-ctx{cfg['context']}-h{GROUPS[group]['horizon']}"


def collect(device: str, only: list[str] | None) -> None:
    for group, g in GROUPS.items():
        if only and group not in only:
            continue
        span = g["n_origins"] * g["stride"]
        for cfg in g["configs"]:
            for period in ("val", "test"):
                out = OUT / group / run_id(group, period, cfg)
                if (out / "done.json").exists():
                    print(f"skip {group}/{out.name}", flush=True)
                    continue
                out.mkdir(parents=True, exist_ok=True)
                model = get_model(cfg["model"], device)
                started = time.time()
                for symbol in g["symbols"]:
                    if (out / f"{symbol}.npz").exists():
                        continue
                    close = load_bars(symbol, g["interval"])
                    need = cfg["context"] + 2 * span + g["horizon"]
                    if len(close) < need:
                        raise SystemExit(f"{symbol} {g['interval']}: {len(close)} bars, need {need}")
                    end = len(close) if period == "test" else len(close) - span
                    c = close[:end]
                    params = BacktestParams(
                        context=cfg["context"], horizon=g["horizon"], stride=g["stride"],
                        max_origins=g["n_origins"] + 1,  # the newest origin has nothing realized and is dropped
                        batch_size=32,
                    )
                    bt = run_backtest(model, c, params, raw=True)
                    arrays = evaluate(c, bt["raw"], g["horizon"])
                    tmp = out / f".{symbol}.tmp.npz"
                    np.savez_compressed(tmp, **arrays)
                    os.replace(tmp, out / f"{symbol}.npz")
                    skill = 1 - arrays["crps"].sum() / arrays["naive_crps"].sum()
                    print(f"  {group}/{out.name} {symbol}: CRPS skill {skill * 100:+.2f}% "
                          f"({len(arrays['index'])} origins)", flush=True)
                (out / "done.json").write_text(json.dumps(
                    {"group": group, "period": period, **cfg, "interval": g["interval"], "horizon": g["horizon"],
                     "stride": g["stride"], "symbols": g["symbols"], "seconds": round(time.time() - started, 1)}
                ))
                print(f"done {group}/{out.name} in {time.time() - started:.0f}s", flush=True)


def load_run(path: Path, symbols: list[str]) -> dict[str, dict[str, np.ndarray]]:
    return {s: dict(np.load(path / f"{s}.npz")) for s in symbols}


def pooled(group: str, run: dict, horizons, block: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    resamples = {s: block_indices(len(a["index"]), block, rng, N_BOOT) for s, a in run.items()}
    out = {}
    for h in horizons:
        m = metrics(run, h)
        b = boot_skill(run, h, resamples)
        m["crps_skill_ci"] = [float(np.percentile(b, 5)), float(np.percentile(b, 95))]
        # per-symbol CIs with the same resamples
        m["per_pair_ci"] = {}
        for s in run:
            bs = boot_skill({s: run[s]}, h, {s: resamples[s]})
            m["per_pair_ci"][s] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
        out[h] = m
    return out


def evaluate_all() -> None:
    pct = lambda v: f"{v * 100:+.2f}"  # noqa: E731
    report = {}
    lines = ["CRPS skill vs the random walk, % (90% block-bootstrap CI). Positive: Timestral is better.", ""]
    for group, g in GROUPS.items():
        block = math.ceil(g["horizon"] / g["stride"])
        hs = g["horizons"]
        lines += [f"## {group}", "",
                  "| run | " + " | ".join(f"h={h}" for h in hs) + f" | MAE skill h={hs[-2]} | cov h={hs[-2]} "
                  f"| calib err | dir h={hs[-2]} | width h={hs[-2]} |",
                  "|---|" + "---|" * (len(hs) + 5)]
        per_pair_lines = []
        for cfg in g["configs"]:
            for period in ("val", "test"):
                rid = run_id(group, period, cfg)
                path = OUT / group / rid
                if not (path / "done.json").exists():
                    lines.append(f"| {rid} | not finished |")
                    continue
                run = load_run(path, g["symbols"])
                # USD/TRY's managed devaluation dominates any pool it is in, so also pool without it
                variants = [(rid, run)]
                if "TRY=X" in run:
                    variants.append((f"{rid} without USD/TRY", {s: a for s, a in run.items() if s != "TRY=X"}))
                for label, rr in variants:
                    res = pooled(group, rr, hs, block)
                    report[f"{group}/{label}"] = res
                    cells = [f"{pct(res[h]['crps_skill'])} ({pct(res[h]['crps_skill_ci'][0])}, "
                             f"{pct(res[h]['crps_skill_ci'][1])})" for h in hs]
                    m = res[hs[-2]]
                    lines.append(f"| {label} | " + " | ".join(cells) + f" | {pct(m['mae_skill'])} | "
                                 f"{m['coverage'] * 100:.1f} | {m['calibration_error'] * 100:.2f} | "
                                 f"{m['direction_hit'] * 100:.1f} | {m['width'] * 100:.2f} |")
                    if cfg is g["configs"][0] and label == rid:
                        per_pair_lines.append((period, res[hs[-2]]))
        lines += ["", f"Per symbol, main config, CRPS skill at h={hs[-2]} (90% CI):", "",
                  "| symbol | " + " | ".join(p for p, _ in per_pair_lines) + " |",
                  "|---|" + "---|" * len(per_pair_lines)]
        for s in g["symbols"]:
            lines.append(f"| {s} | " + " | ".join(
                f"{pct(m['per_pair'][s])} ({pct(m['per_pair_ci'][s][0])}, {pct(m['per_pair_ci'][s][1])})"
                for _, m in per_pair_lines) + " |")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n")
    print(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--device", default="cpu", help="cpu, cuda or mps (mps: transformer on the GPU)")
    c.add_argument("--groups", nargs="*", choices=list(GROUPS), help="default: all")
    sub.add_parser("evaluate")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args.device, args.groups)
    else:
        evaluate_all()


if __name__ == "__main__":
    main()
