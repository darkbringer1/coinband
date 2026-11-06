"""SVG figures for part two of the write-up, from `results/phase1/summary.json` (run `risk.py` first).

    uv run python experiments/phase1/figures.py

Writes to docs/figures/: risk-crps-vs-garch.svg, risk-var.svg, risk-equity.svg. Plain SVG, no dependency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "experiments" / "results" / "phase1" / "summary.json"
OUT = ROOT / "docs" / "figures"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FONT = 'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif" font-size="12"'


def svg(w: int, h: int, body: list[str]) -> str:
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT}>',
                      f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>', *body, "</svg>"]) + "\n"


def title(text: str, sub: str) -> list[str]:
    return [f'<text x="24" y="28" font-size="15" font-weight="600" fill="{INK}">{text}</text>',
            f'<text x="24" y="46" fill="{MUTED}">{sub}</text>']


def y_axis(x0, x1, Y, lo, hi, step, fmt) -> list[str]:
    out = []
    v = math.ceil(lo / step) * step
    while v <= hi + 1e-12:
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}">{fmt(v)}</text>')
        v += step
    return out


def label_column(ends: list[tuple[float, str, str]], x: float) -> list[str]:
    """Direct labels at line ends, nudged apart so they never overlap."""
    out, used = [], []
    for y, text, color in sorted(ends):
        while any(abs(y - u) < 14 for u in used):
            y += 2
        used.append(y)
        out.append(f'<circle cx="{x - 6}" cy="{y - 4:.1f}" r="3" fill="{color}"/>')
        out.append(f'<text x="{x}" y="{y:.1f}" fill="{INK}">{text}</text>')
    return out


def crps_curves(s: dict) -> None:
    w, h, top, bottom = 980, 380, 92, 316
    series = [("timestral", "Timestral", BLUE), ("blend", "Blend (Timestral + GARCH)", ORANGE)]
    vals = [v for p in s["periods"].values() for key, _, _ in series for v in p["curves"][key]]
    lo, hi = min(-0.01, min(vals)) - 0.002, max(vals) + 0.003
    body = title("CRPS skill vs GARCH(1,1), by hours ahead",
                 "Eight Binance pairs, a new forecast every hour, ~4,500 per pair and period. Above 0: better than GARCH.")
    for i, (period, label) in enumerate((("tuning", "Tuning half, Oct 2025 to May 2026"), ("heldout", "Held-out half, May to Nov 2026"))):
        x0 = 70 + i * 470
        x1 = x0 + 250
        X = lambda k: x0 + (k - 1) / 23 * (x1 - x0)  # noqa: E731
        Y = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)  # noqa: E731
        body.append(f'<text x="{x0}" y="{top - 16}" font-weight="600" fill="{INK}">{label}</text>')
        body += y_axis(x0, x1, Y, lo, hi, 0.005, lambda v: f"{v * 100:+.1f}%")
        body.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="{MUTED}" stroke-dasharray="4 4"/>')
        body.append(f'<text x="{x0 + 4}" y="{Y(0) + 15:.1f}" fill="{MUTED}">GARCH = 0</text>')
        for k in (1, 6, 12, 18, 24):
            body.append(f'<text x="{X(k):.1f}" y="{bottom + 18}" text-anchor="middle" fill="{MUTED}">{k}</text>')
        body.append(f'<text x="{x1}" y="{bottom + 34}" text-anchor="end" fill="{MUTED}">hours ahead</text>')
        ends = []
        for key, name, color in series:
            curve = s["periods"][period]["curves"][key]
            pts = " ".join(f"{X(k + 1):.1f},{Y(v):.1f}" for k, v in enumerate(curve))
            body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
            ends.append((Y(curve[-1]) + 4, name, color))
        body += label_column(ends, x1 + 14)
    (OUT / "risk-crps-vs-garch.svg").write_text(svg(w, h, body))


def var_chart(s: dict) -> None:
    w, h = 980, 316
    models = [("timestral", "Timestral"), ("blend", "Blend"), ("garch", "GARCH"), ("ewma", "EWMA"),
              ("hs", "Historical sim."), ("rw", "Random walk")]
    body = title("Value-at-Risk calibration, 1 hour ahead, held-out half",
                 "How often the price fell below each model's 1% and 5% line. On the target line is right; the count is how many of 8 pairs the Kupiec test rejects.")
    panels = [("1%@1h", 0.01, (0.0, 0.025), "1% line (worst hour in a hundred)"), ("5%@1h", 0.05, (0.03, 0.065), "5% line (worst hour in twenty)")]
    top, row = 96, 34
    for i, (key, target, (lo, hi), label) in enumerate(panels):
        x0 = 150 + i * 420
        x1 = x0 + 230
        X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        body.append(f'<text x="{x0}" y="{top - 20}" font-weight="600" fill="{INK}">{label}</text>')
        step = 0.005
        v = lo
        while v <= hi + 1e-12:
            body.append(f'<line x1="{X(v):.1f}" y1="{top - 6}" x2="{X(v):.1f}" y2="{top + row * len(models) - 10}" stroke="{GRID}"/>')
            body.append(f'<text x="{X(v):.1f}" y="{top + row * len(models) + 6}" text-anchor="middle" fill="{MUTED}">{v * 100:.1f}%</text>')
            v += step
        body.append(f'<line x1="{X(target):.1f}" y1="{top - 10}" x2="{X(target):.1f}" y2="{top + row * len(models) - 6}" stroke="{INK}" stroke-width="1.5" stroke-dasharray="5 3"/>')
        body.append(f'<text x="{X(target) + 4:.1f}" y="{top - 8}" fill="{INK}">target</text>')
        for j, (m, name) in enumerate(models):
            y = top + 8 + j * row
            r = s["periods"]["heldout"]["var"][m][key]
            if i == 0:
                body.append(f'<text x="{x0 - 16}" y="{y + 4}" text-anchor="end" fill="{INK}">{name}</text>')
            color = BLUE if m in ("timestral", "blend") else MUTED
            hr = min(max(r["hit_rate"], lo), hi)
            body.append(f'<line x1="{X(target):.1f}" y1="{y}" x2="{X(hr):.1f}" y2="{y}" stroke="{color}" stroke-opacity="0.35" stroke-width="3"/>')
            body.append(f'<circle cx="{X(hr):.1f}" cy="{y}" r="5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
            rej = r["kupiec_rejections"]
            body.append(f'<text x="{x1 + 10}" y="{y + 4}" fill="{MUTED}">{r["hit_rate"] * 100:.2f}% · {rej}/8 rejected</text>')
    (OUT / "risk-var.svg").write_text(svg(w, h, body))


def equity(s: dict) -> None:
    w, h, top, bottom = 960, 380, 80, 320
    series = [("buy_and_hold", "Buy and hold", MUTED), ("garch", "Sized by GARCH", ORANGE),
              ("blend", "Sized by the blend", AQUA), ("timestral", "Sized by Timestral", BLUE)]
    eq = s["periods"]["heldout"]["equity"]
    vals = [math.exp(v) - 1 for key, _, _ in series for v in eq[key]]
    lo, hi = min(vals) - 0.03, max(vals) + 0.03
    n = len(eq["buy_and_hold"])
    x0, x1 = 70, 720
    X = lambda i: x0 + i / (n - 1) * (x1 - x0)  # noqa: E731
    Y = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)  # noqa: E731
    body = title("Volatility targeting, held-out half (May to Nov 2026)",
                 "Equal-weight basket of the eight pairs, resized daily to a fixed risk level, after 10 bp fees. Differences are not statistically significant.")
    body += y_axis(x0, x1, Y, lo, hi, 0.1, lambda v: f"{v * 100:+.0f}%")
    body.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="{MUTED}"/>')
    for frac, text in ((0, "May"), (0.5, "Aug"), (1, "Nov")):
        body.append(f'<text x="{X(frac * (n - 1)):.1f}" y="{bottom + 18}" text-anchor="middle" fill="{MUTED}">{text} 2026</text>')
    ends = []
    for key, name, color in series:
        pts = " ".join(f"{X(i):.1f},{Y(math.exp(v) - 1):.1f}" for i, v in enumerate(eq[key]))
        width = 2.5 if key == "timestral" else 2
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        final = math.exp(eq[key][-1]) - 1
        ends.append((Y(final) + 4, f"{name} {final * 100:+.0f}%", color))
    body += label_column(ends, x1 + 20)
    (OUT / "risk-equity.svg").write_text(svg(w, h, body))


def main() -> None:
    s = json.loads(SRC.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    crps_curves(s)
    var_chart(s)
    equity(s)
    print("wrote risk-crps-vs-garch.svg, risk-var.svg, risk-equity.svg")


if __name__ == "__main__":
    main()
