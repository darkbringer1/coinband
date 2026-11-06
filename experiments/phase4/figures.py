"""SVG figures for part four, from scores only (no Yahoo prices): `results/phase4/risk/summary.json`,
`results/phase4/risk/vix.json` and, for the crypto row, `results/phase1/summary.json`.

    uv run python experiments/phase4/figures.py   # writes docs/figures/markets-vs-garch.svg and markets-vix.svg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase1"))
from figures import BLUE, GRID, INK, MUTED, ORANGE, svg, title  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RISK = ROOT / "experiments" / "results" / "phase4" / "risk"
CRYPTO = ROOT / "experiments" / "results" / "phase1" / "summary.json"
OUT = ROOT / "docs" / "figures"
ROWS = [("crypto", "Crypto, hourly (part 2)"), ("fx-1h", "FX and gold, hourly"),
        ("stocks-1h", "Stocks and indices, hourly"), ("daily", "Everything, daily")]


def vs_garch_chart() -> None:
    risk = json.loads((RISK / "summary.json").read_text())
    crypto = json.loads(CRYPTO.read_text())

    def get(group: str, period: str, key: str) -> dict:
        src = crypto["periods"] if group == "crypto" else risk[group]["periods"]
        return src[period]["crps_skill"][key]

    w, row, top = 960, 50, 118
    h = top + row * len(ROWS) + 76
    lo, hi = -0.006, 0.014
    body = title("Timestral against GARCH, the standard risk model, in four markets",
                 "CRPS skill vs GARCH (whole forecast distribution), average over the horizon, with 90% intervals. "
                 "Right of 0: better than GARCH.")
    panels = (("timestral_vs_garch", "Timestral alone", 300), ("blend_vs_garch", "Blend (Timestral + GARCH)", 640))
    for key, label, x0 in panels:
        x1 = x0 + 270
        X = lambda v, x0=x0, x1=x1: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        body.append(f'<text x="{x0}" y="{top - 34}" font-weight="600" fill="{INK}">{label}</text>')
        v = -0.005
        while v <= hi + 1e-9:
            body.append(f'<line x1="{X(v):.1f}" y1="{top - 16}" x2="{X(v):.1f}" y2="{top + row * len(ROWS) - 14}" stroke="{GRID}"/>')
            body.append(f'<text x="{X(v):.1f}" y="{top + row * len(ROWS) + 4}" text-anchor="middle" fill="{MUTED}">{v * 100:+.1f}%</text>')
            v += 0.005
        body.append(f'<line x1="{X(0):.1f}" y1="{top - 16}" x2="{X(0):.1f}" y2="{top + row * len(ROWS) - 14}" stroke="{INK}" stroke-width="1.5"/>')
        for j, (group, name) in enumerate(ROWS):
            y = top + j * row
            if key == panels[0][0]:
                body.append(f'<text x="{x0 - 20}" y="{y + 4}" text-anchor="end" fill="{INK}">{name}</text>')
            for period, color, dy in (("tuning", ORANGE, -7), ("heldout", BLUE, 7)):
                r = get(group, period, key)
                a, b = r["ci"]
                yy = y + dy
                body.append(f'<line x1="{X(a):.1f}" y1="{yy}" x2="{X(b):.1f}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
                body.append(f'<circle cx="{X(r["avg"]):.1f}" cy="{yy}" r="5" fill="{color}" stroke="#fcfcfb" stroke-width="2"/>')
    ly = top + row * len(ROWS) + 40
    for k, (name, color) in enumerate((("Tuning half (GARCH fitted here)", ORANGE), ("Held-out half", BLUE))):
        lx = 300 + k * 300
        body.append(f'<circle cx="{lx}" cy="{ly - 4}" r="5" fill="{color}"/>')
        body.append(f'<text x="{lx + 10}" y="{ly}" fill="{INK}">{name}</text>')
    (OUT / "markets-vs-garch.svg").write_text(svg(w, h, body))


def vix_chart() -> None:
    s = json.loads((RISK / "vix.json").read_text())
    names = ["implied", "GARCH", "Timestral", "Blend"]
    labels = {"implied": "Options market (VIX / VXN)", "GARCH": "GARCH", "Timestral": "Timestral",
              "Blend": "Blend (Timestral + GARCH)"}
    w, h, top = 960, 330, 100
    lo, hi = 0.0, 0.5
    n = s["^GSPC"]["n_heldout"]
    body = title("Who forecasts next month's stock market volatility best?",
                 f"Share of the variation in realized 21-day volatility explained (held-out R², log scale), {n} trading "
                 "days from Oct 2020 to Oct 2026. Higher is better.")
    for i, (sym, name) in enumerate((("^GSPC", "S&amp;P 500 vs VIX"), ("^NDX", "Nasdaq-100 vs VXN"))):
        x0 = 250 + i * 350
        x1 = x0 + 250
        X = lambda v, x0=x0, x1=x1: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        body.append(f'<text x="{x0}" y="{top - 22}" font-weight="600" fill="{INK}">{name}</text>')
        v = 0.0
        while v <= hi + 1e-9:
            body.append(f'<line x1="{X(v):.1f}" y1="{top - 10}" x2="{X(v):.1f}" y2="{top + 4 * 44 - 10}" stroke="{GRID}"/>')
            body.append(f'<text x="{X(v):.1f}" y="{top + 4 * 44 + 6}" text-anchor="middle" fill="{MUTED}">{v:.1f}</text>')
            v += 0.1
        for j, nm in enumerate(names):
            y = top + 8 + j * 44
            if i == 0:
                body.append(f'<text x="{x0 - 16}" y="{y + 4}" text-anchor="end" fill="{INK}">{labels[nm]}</text>')
            r2 = s[sym]["forecasts"][nm]["r2_heldout"]
            color = BLUE if nm in ("Timestral", "Blend") else MUTED
            body.append(f'<rect x="{X(0):.1f}" y="{y - 9}" width="{max(X(r2) - X(0), 1):.1f}" height="18" rx="3" fill="{color}"/>')
            body.append(f'<text x="{X(r2) + 6:.1f}" y="{y + 4}" fill="{INK}">{r2:.2f}</text>')
    (OUT / "markets-vix.svg").write_text(svg(w, h, body))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vs_garch_chart()
    vix_chart()
    print("wrote markets-vs-garch.svg, markets-vix.svg")


if __name__ == "__main__":
    main()
