"""Paired effects for the write-up: change one setting, keep the rest, on the same origins.

Prints, for each period, the change in pooled CRPS skill (percentage points) with a 90% moving-block
bootstrap interval at horizons 1, 6, 24 and 96, plus per-pair CRPS skill of the chosen config, and the
CRPS skill curve over every horizon step (for the figure).

    uv run python experiments/effects.py
"""

from __future__ import annotations

import json
import math

import numpy as np

from analyze import HORIZONS, N_BOOT, RESULTS, STRIDE, block_indices, boot_skill, load_run, metrics

STAGE = RESULTS / "confirm"
CHOSEN = "1h-209M-ctx4096-cov0-trend0-h96"
EFFECTS = [
    ("volume covariates, 209M ctx4096", "1h-209M-ctx4096-cov0-trend0-h96", "1h-209M-ctx4096-cov1-trend0-h96"),
    ("volume covariates, 28M ctx4096", "1h-28M-ctx4096-cov0-trend0-h96", "1h-28M-ctx4096-cov1-trend0-h96"),
    ("context 2048 to 4096, 209M", "1h-209M-ctx2048-cov0-trend0-h96", "1h-209M-ctx4096-cov0-trend0-h96"),
    ("context 4096 to 8192, 209M", "1h-209M-ctx4096-cov0-trend0-h96", "1h-209M-ctx8192-cov0-trend0-h96"),
    ("context 2048 to 4096, 28M", "1h-28M-ctx2048-cov0-trend0-h96", "1h-28M-ctx4096-cov0-trend0-h96"),
    ("28M to 123M, ctx4096", "1h-28M-ctx4096-cov0-trend0-h96", "1h-123M-ctx4096-cov0-trend0-h96"),
    ("123M to 209M, ctx4096", "1h-123M-ctx4096-cov0-trend0-h96", "1h-209M-ctx4096-cov0-trend0-h96"),
    ("28M ctx2048 (old default) to 209M ctx4096", "1h-28M-ctx2048-cov0-trend0-h96", "1h-209M-ctx4096-cov0-trend0-h96"),
]


def curve(run: dict, horizon: int = 96) -> list[float]:
    """CRPS skill at each single step k (not cumulative), pooled over pairs."""
    num = sum(a["crps"].sum(axis=0) for a in run.values())
    den = sum(a["naive_crps"].sum(axis=0) for a in run.values())
    return [float(v) for v in (1 - num / den)[:horizon]]


def main() -> None:
    out = {}
    for period in ("val", "test"):
        runs = {name[len(period) + 1 :]: load_run(STAGE / name) for name in
                (p.name for p in STAGE.iterdir() if p.name.startswith(period + "-"))}
        ref = runs[CHOSEN]
        rng = np.random.default_rng(1)
        resamples = {s: block_indices(len(a["index"]), math.ceil(max(HORIZONS) / STRIDE), rng, N_BOOT) for s, a in ref.items()}
        print(f"\n== {period}: change in CRPS skill, points (90% CI)")
        print("| change | h=1 | h=6 | h=24 | h=96 |\n|---|---|---|---|---|")
        rows = []
        for label, a, b in EFFECTS:
            cells = []
            for h in HORIZONS:
                d = boot_skill(runs[b], h, resamples) - boot_skill(runs[a], h, resamples)
                point = metrics(runs[b], h)["crps_skill"] - metrics(runs[a], h)["crps_skill"]
                lo, hi = np.percentile(d, 5), np.percentile(d, 95)
                cells.append(f"{point * 100:+.2f} ({lo * 100:+.2f}, {hi * 100:+.2f})")
            rows.append({"change": label, "cells": cells})
            print(f"| {label} | " + " | ".join(cells) + " |")
        per_pair = metrics(ref, 24)["per_pair"]
        print(f"\nper-pair CRPS skill h=24, {CHOSEN}: " + ", ".join(f"{s} {v * 100:+.2f}" for s, v in per_pair.items()))
        old = metrics(runs["1h-28M-ctx2048-cov0-trend0-h96"], 24)["per_pair"]
        print("old default per pair: " + ", ".join(f"{s} {v * 100:+.2f}" for s, v in old.items()))
        out[period] = {
            "effects": rows,
            "per_pair": per_pair,
            "per_pair_old": old,
            "curve": curve(ref),
            "curve_old": curve(runs["1h-28M-ctx2048-cov0-trend0-h96"]),
        }
    (STAGE / "effects.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
