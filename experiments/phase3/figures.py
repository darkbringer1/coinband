"""SVG figures for part three, from `results/phase2/summary.json` and `results/phase3/summary.json`.

    uv run python experiments/phase3/figures.py   # writes docs/figures/inputs.svg and options-r2.svg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase1"))
from figures import BLUE, GRID, INK, MUTED, ORANGE, svg, title  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
P2 = ROOT / "experiments" / "results" / "phase2" / "summary.json"
P3 = ROOT / "experiments" / "results" / "phase3" / "summary.json"
OUT = ROOT / "docs" / "figures"
LABELS = {"market": "Market leader (BTC/ETH)", "funding": "Funding rate", "calendar": "Hour and weekday",
          "volume": "Volume", "all": "All of them"}


def inputs_chart() -> None:
    s = json.loads(P2.read_text())
    configs = [c for c in LABELS if c in s["tuning"]]
    w, row, top = 960, 46, 100
    h = top + row * len(configs) + 70
    lo, hi = -0.012, 0.016
    x0, x1 = 250, 700
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    body = title("What extra inputs do to Timestral's forecast ranges",
                 "CRPS skill vs plain Timestral (price only), average over 1 to 24 hours ahead, with 90% intervals. Right of 0: the input helps.")
    v = -0.01
    while v <= hi + 1e-9:
        body.append(f'<line x1="{X(v):.1f}" y1="{top - 10}" x2="{X(v):.1f}" y2="{top + row * len(configs) - 10}" stroke="{GRID}"/>')
        body.append(f'<text x="{X(v):.1f}" y="{top + row * len(configs) + 8}" text-anchor="middle" fill="{MUTED}">{v * 100:+.1f}%</text>')
        v += 0.005
    body.append(f'<line x1="{X(0):.1f}" y1="{top - 14}" x2="{X(0):.1f}" y2="{top + row * len(configs) - 6}" stroke="{INK}" stroke-width="1.5"/>')
    body.append(f'<text x="{X(0) + 4:.1f}" y="{top - 18}" fill="{INK}">no change</text>')
    for j, c in enumerate(configs):
        y = top + 10 + j * row
        body.append(f'<text x="{x0 - 16}" y="{y + 4}" text-anchor="end" fill="{INK}">{LABELS[c]}</text>')
        for k, (period, color, dy) in enumerate((("tuning", ORANGE, -7), ("heldout", BLUE, 7))):
            r = s[period][c]
            lo_ci, hi_ci = r["vs_plain_ci"]
            yy = y + dy
            body.append(f'<line x1="{X(lo_ci):.1f}" y1="{yy}" x2="{X(hi_ci):.1f}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
            body.append(f'<circle cx="{X(r["vs_plain"]):.1f}" cy="{yy}" r="5" fill="{color}" stroke="#fcfcfb" stroke-width="2"/>')
            body.append(f'<text x="{x1 + 14}" y="{yy + 4}" fill="{MUTED}">{"tuning" if k == 0 else "held-out"} {r["vs_plain"] * 100:+.2f}%</text>')
    ly = top + row * len(configs) + 34
    for k, (name, color) in enumerate((("Tuning half (Oct 2025 to May 2026)", ORANGE), ("Held-out half (May to Nov 2026)", BLUE))):
        lx = 250 + k * 300
        body.append(f'<circle cx="{lx}" cy="{ly - 4}" r="5" fill="{color}"/>')
        body.append(f'<text x="{lx + 10}" y="{ly}" fill="{INK}">{name}</text>')
    (OUT / "inputs.svg").write_text(svg(w, h, body))


def options_chart() -> None:
    s = json.loads(P3.read_text())
    names = ["DVOL (implied)", "GARCH", "Timestral", "Blend"]
    labels = {"DVOL (implied)": "Options market (DVOL)", "GARCH": "GARCH", "Timestral": "Timestral", "Blend": "Blend (Timestral + GARCH)"}
    w, h, top = 960, 330, 100
    lo, hi = -0.2, 0.25
    body = title("Who forecasts tomorrow's volatility best?",
                 "Share of the day-to-day variation in realized 24 h volatility explained (held-out R², log scale), 188 days. Higher is better; below 0 is worse than a constant.")
    for i, sym in enumerate(("BTCUSDT", "ETHUSDT")):
        x0 = 230 + i * 360
        x1 = x0 + 260
        X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        body.append(f'<text x="{x0}" y="{top - 22}" font-weight="600" fill="{INK}">{sym[:3]}</text>')
        v = -0.2
        while v <= hi + 1e-9:
            body.append(f'<line x1="{X(v):.1f}" y1="{top - 10}" x2="{X(v):.1f}" y2="{top + 4 * 44 - 10}" stroke="{GRID}"/>')
            body.append(f'<text x="{X(v):.1f}" y="{top + 4 * 44 + 6}" text-anchor="middle" fill="{MUTED}">{v:.1f}</text>')
            v += 0.1
        body.append(f'<line x1="{X(0):.1f}" y1="{top - 10}" x2="{X(0):.1f}" y2="{top + 4 * 44 - 10}" stroke="{INK}"/>')
        for j, n in enumerate(names):
            y = top + 8 + j * 44
            if i == 0:
                body.append(f'<text x="{x0 - 16}" y="{y + 4}" text-anchor="end" fill="{INK}">{labels[n]}</text>')
            r2 = s[sym]["forecasts"][n]["r2_heldout"]
            color = BLUE if n in ("Timestral", "Blend") else MUTED
            a, b = sorted((X(0), X(max(lo, r2))))
            body.append(f'<rect x="{a:.1f}" y="{y - 9}" width="{max(b - a, 1):.1f}" height="18" rx="3" fill="{color}"/>')
            anchor, tx = ("start", b + 6) if r2 >= 0 else ("end", a - 6)
            body.append(f'<text x="{tx:.1f}" y="{y + 4}" text-anchor="{anchor}" fill="{INK}">{r2:+.2f}</text>')
    (OUT / "options-r2.svg").write_text(svg(w, h, body))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs_chart()
    options_chart()
    print("wrote inputs.svg, options-r2.svg")


if __name__ == "__main__":
    main()
