"""Tables for the write-up from the arrays `sweep.py` saved.

For every run and horizon h (scores over steps 1..h, direction at step h):

- CRPS skill: 1 - sum(CRPS) / sum(naive random-walk CRPS), pooled over pairs (both are normalized by the
  last known price, so no pair dominates by its price level).
- MAE skill of the median vs "price stays flat", 80% coverage, mean calibration error, direction hit rate
  (flat calls and flat moves count as misses), mean 80% band width.
- A 90% confidence interval for CRPS skill and for its difference to a reference run, from a moving-block
  bootstrap over origins (blocks span the horizon so overlapping forecasts stay together; the same
  resamples are used for every run, so differences are paired).

    uv run python experiments/analyze.py --stage screen --ref val-1h-28M-ctx2048-cov0-trend0-h96
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
HORIZONS = (1, 6, 24, 96)
N_BOOT = 1000
STRIDE = 8


def load_run(path: Path) -> dict[str, dict[str, np.ndarray]]:
    return {p.stem: dict(np.load(p)) for p in sorted(path.glob("*.npz"))}


def block_indices(n: int, block: int, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """`(n_boot, n)` moving-block bootstrap resamples of `range(n)`."""
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx


def metrics(run: dict, h: int) -> dict:
    crps = sum(float(a["crps"][:, :h].sum()) for a in run.values())
    naive = sum(float(a["naive_crps"][:, :h].sum()) for a in run.values())
    ae = sum(float(a["ae"][:, :h].sum()) for a in run.values())
    naive_ae = sum(float(a["naive_ae"][:, :h].sum()) for a in run.values())
    inside = np.concatenate([a["inside"][:, :h].ravel() for a in run.values()])
    below = np.concatenate([a["below"][:, :h].reshape(-1, a["below"].shape[-1]) for a in run.values()])
    levels = next(iter(run.values()))["levels"]
    pred = np.concatenate([a["dir_pred"][:, h - 1] for a in run.values()])
    real = np.concatenate([a["dir_real"][:, h - 1] for a in run.values()])
    width = np.concatenate([a["width"][:, :h].ravel() for a in run.values()])
    return {
        "crps_skill": 1 - crps / naive,
        "mae_skill": 1 - ae / naive_ae,
        "coverage": float(inside.mean()),
        "calibration_error": float(np.abs(below.mean(axis=0) - levels).mean()),
        "direction_hit": float(((pred != 0) & (pred == real)).mean()),
        "width": float(width.mean()),
        "per_pair": {
            s: 1 - float(a["crps"][:, :h].sum()) / float(a["naive_crps"][:, :h].sum()) for s, a in run.items()
        },
    }


def boot_skill(run: dict, h: int, resamples: dict[str, np.ndarray]) -> np.ndarray:
    """`(n_boot,)` pooled CRPS skill under the shared resamples."""
    num = 0.0
    den = 0.0
    for s, a in run.items():
        c = a["crps"][:, :h].sum(axis=1)
        nv = a["naive_crps"][:, :h].sum(axis=1)
        idx = resamples[s]
        num = num + c[idx].sum(axis=1)
        den = den + nv[idx].sum(axis=1)
    return 1 - num / den


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--ref", required=True, help="run id the others are compared against")
    ap.add_argument("--period", default=None, help="only runs of this period (val/test)")
    args = ap.parse_args(argv)

    stage_dir = RESULTS / args.stage
    runs = {p.name: load_run(p) for p in sorted(stage_dir.iterdir()) if (p / "done.json").exists()}
    if args.period:
        runs = {k: v for k, v in runs.items() if k.startswith(args.period + "-")}
    if args.ref not in runs:
        raise SystemExit(f"reference run {args.ref} not finished")
    ref = runs[args.ref]
    rng = np.random.default_rng(0)
    block = math.ceil(max(HORIZONS) / STRIDE)
    resamples = {s: block_indices(len(a["index"]), block, rng, N_BOOT) for s, a in ref.items()}

    table = []
    for name, run in runs.items():
        if set(run) != set(ref) or any(len(run[s]["index"]) != len(ref[s]["index"]) for s in ref):
            print(f"skip {name}: different pairs or origins than the reference")
            continue
        row = {"run": name, "h": {}}
        for h in HORIZONS:
            m = metrics(run, h)
            b = boot_skill(run, h, resamples)
            d = b - boot_skill(ref, h, resamples)
            m["crps_skill_ci"] = [float(np.percentile(b, 5)), float(np.percentile(b, 95))]
            m["delta_vs_ref"] = m["crps_skill"] - metrics(ref, h)["crps_skill"]
            m["delta_ci"] = [float(np.percentile(d, 5)), float(np.percentile(d, 95))]
            row["h"][h] = m
        table.append(row)

    table.sort(key=lambda r: -r["h"][24]["crps_skill"])
    (stage_dir / f"summary-{args.period or 'all'}.json").write_text(json.dumps(table, indent=1))

    pct = lambda v: f"{v * 100:+.2f}"  # noqa: E731
    lines = [
        f"Reference: `{args.ref}`. CRPS skill vs random walk, % (90% CI). Δ = paired difference to the reference.",
        "",
        "| run | h=1 | h=6 | h=24 | Δ h=24 (90% CI) | h=96 | MAE skill h=24 | cov h=24 | calib err | dir h=24 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        h = r["h"]
        m24 = h[24]
        lo, hi = m24["delta_ci"]
        lines.append(
            f"| {r['run']} | {pct(h[1]['crps_skill'])} | {pct(h[6]['crps_skill'])} | {pct(m24['crps_skill'])} "
            f"| {pct(m24['delta_vs_ref'])} ({pct(lo)}, {pct(hi)}) | {pct(h[96]['crps_skill'])} "
            f"| {pct(m24['mae_skill'])} | {m24['coverage'] * 100:.1f} | {m24['calibration_error'] * 100:.2f} "
            f"| {m24['direction_hit'] * 100:.1f} |"
        )
    out = "\n".join(lines)
    (stage_dir / f"summary-{args.period or 'all'}.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
