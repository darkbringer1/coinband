"""SVG figure for the write-up, from `results/confirm/effects.json` (run `effects.py` first).

CRPS skill against the random walk at each step ahead, val and test periods side by side, for the chosen
config and the old default. Plain SVG, no plotting dependency.

    uv run python experiments/figures.py   # writes docs/figures/skill-by-horizon.svg and cone-anatomy.svg

`cone-anatomy.svg` is an illustration (a synthetic price path and an idealized cone), not model output.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = Path(__file__).parent / "results" / "confirm" / "effects.json"
OUT = ROOT / "docs" / "figures" / "skill-by-horizon.svg"
ANATOMY = ROOT / "docs" / "figures" / "cone-anatomy.svg"

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


def anatomy() -> None:
    """Annotated sketch of a forecast cone for readers new to quantile forecasts."""
    import math
    import random

    w, h = 960, 380
    x0, x_now, x1 = 40, 400, 690
    y_mid = 200
    rng = random.Random(7)
    # synthetic past price: a gentle random walk ending at y_mid
    pts, y = [], y_mid + 30
    n_past = 60
    for i in range(n_past + 1):
        y += rng.gauss(0, 6) - (y - y_mid) * 0.05
        pts.append((x0 + i / n_past * (x_now - x0), y))
    pts[-1] = (x_now, y_mid)
    steps = 40
    xs = [x_now + k / steps * (x1 - x_now) for k in range(1, steps + 1)]
    spread = [math.sqrt(k / steps) for k in range(1, steps + 1)]
    band = lambda z: [(x, y_mid - z * 120 * s) for x, s in zip(xs, spread)]  # noqa: E731
    p95, p90, p10, p05 = band(1.645 / 1.645), band(1.28 / 1.645), band(-1.28 / 1.645), band(-1.0)
    poly = lambda top, bot: " ".join(f"{x:.1f},{y:.1f}" for x, y in [(x_now, y_mid)] + top + bot[::-1])  # noqa: E731
    future = [(x_now, y_mid)]
    y = y_mid
    for x in xs:
        y += rng.gauss(0, 7)
        future.append((x, y))
    path = lambda p: " ".join(f"{x:.1f},{y:.1f}" for x, y in p)  # noqa: E731
    blue, orange = "#2a78d6", "#eb6834"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif" font-size="13">',
        f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
        f'<text x="24" y="30" font-size="15" font-weight="600" fill="{INK}">Anatomy of a forecast cone</text>',
        f'<text x="24" y="50" fill="{MUTED}">Illustration. Everything left of "now" is known; the cone is the forecast for each hour ahead.</text>',
        f'<polygon points="{poly(p95, p05)}" fill="{blue}" fill-opacity="0.16"/>',
        f'<polygon points="{poly(p90, p10)}" fill="{blue}" fill-opacity="0.32"/>',
        f'<polyline points="{path(future)}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="3 3"/>',
        f'<polyline points="{path(pts)}" fill="none" stroke="{INK}" stroke-width="2"/>',
        f'<line x1="{x_now}" y1="{y_mid}" x2="{x1}" y2="{y_mid}" stroke="{orange}" stroke-width="2" stroke-dasharray="6 4"/>',
        f'<line x1="{x_now}" y1="70" x2="{x_now}" y2="{h - 40}" stroke="{MUTED}" stroke-dasharray="2 4"/>',
        f'<circle cx="{x_now}" cy="{y_mid}" r="5" fill="{orange}"/>',
        f'<text x="{x_now - 8}" y="{h - 26}" text-anchor="end" fill="{MUTED}">past (known)</text>',
        f'<text x="{x_now + 8}" y="{h - 26}" fill="{MUTED}">future (forecast)</text>',
        f'<text x="{x_now}" y="66" text-anchor="middle" font-weight="600" fill="{INK}">now</text>',
    ]
    notes = [
        (p90[-1][1] + 14, "80% band: the price should end up", "in here about 8 times out of 10", blue),
        (p95[-1][1] - 6, "90% band", "(lighter, wider)", blue),
        (y_mid + 4, "median: the single most", "likely value", orange),
        (future[-1][1] + 4, "what actually happened", "(dashed), revealed later", MUTED),
    ]
    used: list[float] = []
    for y_note, a, b, color in sorted(notes, key=lambda n: n[0]):
        y_note = max([y_note] + [u + 34 for u in used])
        used.append(y_note)
        parts.append(f'<text x="{x1 + 10}" y="{y_note:.1f}" fill="{color}" font-weight="600">{a}</text>')
        parts.append(f'<text x="{x1 + 10}" y="{y_note + 15:.1f}" fill="{MUTED}">{b}</text>')
    parts.append("</svg>")
    ANATOMY.write_text("\n".join(parts) + "\n")
    print(f"wrote {ANATOMY}")


if __name__ == "__main__":
    main()
    anatomy()
