"""SVG figure for the write-up, from `results/confirm/effects.json` (run `effects.py` first).

CRPS skill against the random walk at each step ahead, val and test periods side by side, for the chosen
config and the old default. Plain SVG, no plotting dependency.

    uv run python experiments/figures.py   # writes docs/figures/skill-by-horizon.svg
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = Path(__file__).parent / "results" / "confirm" / "effects.json"
OUT = ROOT / "docs" / "figures" / "skill-by-horizon.svg"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#52514e", "#e4e3df"
SERIES = [("Timestral-209M, 4096 context", "#2a78d6", "curve"), ("Timestral-28M, 2048 context (old default)", "#eb6834", "curve_old")]
W, H = 960, 400
PANEL_W, TOP, BOTTOM = 380, 96, 320


def main() -> None:
    data = json.loads(SRC.read_text())
    values = [v for p in data.values() for _, _, key in SERIES for v in p[key]]
    lo = min(-0.01, min(values)) - 0.005
    hi = max(values) + 0.005
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="24" y="28" font-size="15" font-weight="600" fill="{INK}">CRPS skill vs a random walk, by bars ahead</text>',
        f'<text x="24" y="46" fill="{MUTED}">8 Binance pairs, 1h candles, 289 forecasts per pair and period. Above 0 beats the random-walk band.</text>',
    ]
    # legend
    lx = 24
    for name, color, _ in SERIES:
        parts.append(f'<line x1="{lx}" y1="{H - 20}" x2="{lx + 18}" y2="{H - 20}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx + 24}" y="{H - 16}" fill="{INK}">{name}</text>')
        lx += 24 + 7 * len(name) + 32
    for p_i, (period, label) in enumerate((("val", "Tuning period, Apr to Jul 2026"), ("test", "Held-out period, Jul to Nov 2026"))):
        x0 = 70 + p_i * (PANEL_W + 90)
        x1 = x0 + PANEL_W
        X = lambda k: x0 + (k - 1) / 95 * (x1 - x0)  # noqa: E731
        Y = lambda v: BOTTOM - (v - lo) / (hi - lo) * (BOTTOM - TOP)  # noqa: E731
        parts.append(f'<text x="{x0}" y="{TOP - 16}" font-weight="600" fill="{INK}">{label}</text>')
        step = 0.01
        v = (int(lo / step) - 1) * step
        while v <= hi + 1e-9:
            if v >= lo:
                parts.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
                parts.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}">{v * 100:+.0f}%</text>')
            v += step
        parts.append(
            f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="{MUTED}" stroke-dasharray="4 4"/>'
        )
        parts.append(f'<text x="{x0 + 4}" y="{Y(0) + 16:.1f}" fill="{MUTED}">random walk = 0</text>')
        for k in (1, 24, 48, 72, 96):
            parts.append(f'<text x="{X(k):.1f}" y="{BOTTOM + 18}" text-anchor="middle" fill="{MUTED}">{k}</text>')
        parts.append(f'<text x="{x1}" y="{BOTTOM + 34}" text-anchor="end" fill="{MUTED}">bars ahead (hours)</text>')
        for _, color, key in SERIES:
            pts = " ".join(f"{X(k + 1):.1f},{Y(v):.1f}" for k, v in enumerate(data[period][key]))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
