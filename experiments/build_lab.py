"""Build the Lab page data: interactive versions of the strongest results, for GitHub Pages.

From the phase 1 hourly forecasts (`experiments/phase1/collect.py`), for the held-out half and each pair:

- `docs/lab/data/risk-<PAIR>.json`: every hour, the price and the one-hour 1% and 5% Value-at-Risk lines of
  Timestral and of GARCH (fitted on the tuning half), as price levels.
- `docs/lab/data/vol-<PAIR>.json`: every day, the realized volatility of the next 24 hours and the forecasts of
  Timestral, GARCH and the blend, with each day's QLIKE loss.
- `docs/lab/data/index.json`: pairs, dates and headline numbers from `results/phase1/summary.json`.

Only Binance-derived data is written. Deribit DVOL is not published: its terms restrict redistribution.

    uv run python experiments/build_lab.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "phase1"))
from risk import baseline_quantiles, sigma_from_quantiles  # noqa: E402
from sweep import CONFIRM_PAIRS, load_klines  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "docs" / "lab" / "data"
PHASE1 = HERE / "results" / "phase1"
BARS, H = 13152, 24


def r6(a) -> list:
    return [float(f"{v:.6g}") for v in np.asarray(a, dtype=np.float64)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"pairs": [], "generated": datetime.now(UTC).strftime("%Y-%m-%d")}
    for symbol in CONFIRM_PAIRS:
        times, close, _ = load_klines(symbol, "1h", BARS)
        logc = np.log(close)
        f = np.load(PHASE1 / f"{symbol}.npz")
        idx_all, levels = f["index"], f["levels"]
        keep = idx_all + H <= len(close)
        idx_all, q_all = idx_all[keep], f["q"][keep].astype(np.float64)
        half = len(idx_all) // 2
        fit_end = int(idx_all[half])
        idx, q_t = idx_all[half:], q_all[half:]
        q_g = baseline_quantiles(logc, idx, levels, fit_end)["garch"]
        i01, i05 = int(np.argmin(np.abs(levels - 0.01))), int(np.argmin(np.abs(levels - 0.05)))
        last = close[idx - 1]
        risk = {
            "symbol": symbol,
            "t0": int(times[idx[0]]),
            "step": 3_600_000,
            "close": r6(close[idx]),  # the close of the hour each line is for
            "last": r6(last),
            "t01": r6(last * np.exp(q_t[:, 0, i01])),
            "t05": r6(last * np.exp(q_t[:, 0, i05])),
            "g01": r6(last * np.exp(q_g[:, 0, i01])),
            "g05": r6(last * np.exp(q_g[:, 0, i05])),
        }
        (OUT / f"risk-{symbol}.json").write_text(json.dumps(risk, separators=(",", ":")))
        # daily volatility race: one origin per day
        sel = np.arange(0, len(idx), 24)
        d_idx = idx[sel]
        rv = np.array([np.sum(np.diff(logc)[i - 1 : i - 1 + H] ** 2) for i in d_idx])
        sig = {
            "timestral": sigma_from_quantiles(q_t[sel], levels)[:, -1],
            "garch": sigma_from_quantiles(q_g[sel], levels)[:, -1],
        }
        sig["blend"] = sigma_from_quantiles(0.5 * (q_t[sel] + q_g[sel]), levels)[:, -1]
        ql = {k: (rv / v**2 - np.log(rv / v**2) - 1) for k, v in sig.items()}
        ann = math.sqrt(365) * 100  # daily vol -> annualized %
        vol = {
            "symbol": symbol,
            "time": [int(times[i]) for i in d_idx],
            "realized": r6(np.sqrt(rv) * ann),
            **{k: r6(v * ann) for k, v in sig.items()},
            "qlike": {k: r6(v) for k, v in ql.items()},
        }
        (OUT / f"vol-{symbol}.json").write_text(json.dumps(vol, separators=(",", ":")))
        index["pairs"].append(symbol)
        index["start"], index["end"] = int(times[idx[0]]), int(times[idx[-1]])
        print(f"{symbol}: {len(idx)} hours, {len(d_idx)} days", flush=True)
    s = json.loads((PHASE1 / "summary.json").read_text())["periods"]["heldout"]
    index["headline"] = {
        "crps_vs_garch": s["crps_skill"]["timestral_vs_garch"],
        "blend_vs_garch": s["crps_skill"]["blend_vs_garch"],
        "var": {m: s["var"][m]["1%@1h"] | {"5": s["var"][m]["5%@1h"]} for m in ("timestral", "garch", "blend", "ewma")},
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
