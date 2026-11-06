"""Render `docs/paper.md` as a standalone HTML page, with the small Markdown subset it uses.

Headings, paragraphs, bullet and numbered lists, block quotes (callouts), pipe tables, fenced code, images,
links, bold, italics, inline code, and raw HTML lines such as `<details>` / `<summary>` (Markdown inside a
`<details>` block is still rendered, as on GitHub).
Stdlib only, so the app can serve the write-up without a Markdown dependency.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

STYLE = """
:root { color-scheme: light; }
body { margin: 0; background: #fcfcfb; color: #1a1a19;
  font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 24px 80px; }
nav { font-size: 14px; margin-bottom: 24px; }
nav a { color: #2a78d6; }
h1 { font-size: 30px; line-height: 1.25; margin: 0 0 8px; }
h2 { font-size: 22px; margin: 40px 0 12px; }
h3 { font-size: 18px; margin: 28px 0 10px; }
p, li { color: #2b2a28; }
a { color: #2a78d6; }
code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; background: #efeee9; padding: 1px 4px; border-radius: 4px; }
pre { background: #1a1a19; color: #f0efec; padding: 14px 16px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; color: inherit; padding: 0; }
.table-wrap { overflow-x: auto; margin: 12px 0 20px; }
table { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #e4e3df; padding: 6px 10px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; white-space: normal; }
th { color: #52514e; font-weight: 600; }
img { max-width: 100%; height: auto; display: block; margin: 16px 0; }
blockquote { margin: 20px 0; padding: 14px 18px; background: #eef4fc; border-left: 4px solid #2a78d6; border-radius: 6px; }
blockquote p { margin: 6px 0; }
details { margin: 16px 0; border: 1px solid #e4e3df; border-radius: 8px; padding: 0 16px; background: #f6f6f3; }
details[open] { padding-bottom: 8px; }
summary { cursor: pointer; padding: 10px 0; font-weight: 600; color: #3f3e3b; }
details > summary::marker { color: #2a78d6; }
.lede { font-size: 18px; color: #3f3e3b; }
.nav { display: flex; gap: 18px; flex-wrap: wrap; }
"""


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", out)
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(markdown: str, title: str = "coinband paper", nav: str = '<a href="/">&larr; back to coinband</a>') -> str:
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
        f"<style>{STYLE}</style></head><body><main><nav class=\"nav\">{nav}</nav>"
        + render_fragment(markdown)
        + "</main></body></html>"
    )


def render_fragment(markdown: str) -> str:
    lines = markdown.splitlines()
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            body.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            i += 1
            continue
        m = re.match(r"(#{1,6}) (.*)", line)
        if m:
            level = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"[`*]", "", m.group(2)).lower()).strip("-")
            body.append(f'<h{level} id="{slug}">{_inline(m.group(2))}</h{level}>')
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"\|[\s\-|:]+\|$", lines[i + 1].strip()):
            head = _cells(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            trs = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            body.append(f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            continue
        if re.match(r"\s*<", line) and not line.lstrip().startswith("<?"):
            body.append(line.strip())  # raw HTML line (details, summary, div, ...)
            i += 1
            continue
        if line.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i][2:] if lines[i].startswith("> ") else lines[i][1:])
                i += 1
            body.append("<blockquote>" + render_fragment("\n".join(quote)) + "</blockquote>")
            continue
        m = re.match(r"(- |\d+\. )", line)
        if m:
            ordered = m.group(1) != "- "
            items = []
            pattern = r"\d+\. " if ordered else r"- "
            while i < len(lines) and (re.match(pattern, lines[i]) or lines[i].startswith("  ")):
                head = re.match(pattern, lines[i])
                if head:
                    items.append(lines[i][head.end():])
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            body.append(f"<{tag}>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + f"</{tag}>")
            continue
        if not line.strip():
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"(#|- |\d+\. |\||```|>|\s*<)", lines[i]):
            para.append(lines[i])
            i += 1
        body.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(body)


# article name -> (markdown file, URL path in the app)
ARTICLES = {"paper": ("paper.md", "/paper"), "risk": ("risk.md", "/paper/risk")}


def rewrite_links(md: str, base: dict[str, str], figures: str, demo: str) -> str:
    """Point the Markdown's repo-relative links (paper.md, risk.md, figures/, demo/) at where pages are served."""
    md = md.replace("](figures/", f"]({figures}").replace("](demo/)", f"]({demo})")
    for _, (filename, _) in ARTICLES.items():
        name = filename[:-3]
        md = re.sub(rf"\]\({re.escape(filename)}(#[^)]*)?\)", lambda m, n=name: f"]({base[n]}{m.group(1) or ''})", md)
    return md


def paper_html(name: str = "paper") -> str | None:
    if name not in ARTICLES:
        return None
    path = DOCS / ARTICLES[name][0]
    if not path.exists():
        return None
    base = {n: url for n, (_, url) in ARTICLES.items()}
    md = rewrite_links(path.read_text(), base, figures="/paper/figures/", demo="/")  # the app is the demo here
    title = next((ln[2:] for ln in md.splitlines() if ln.startswith("# ")), "coinband paper")
    nav = '<a href="/">&larr; back to coinband</a><a href="/paper">Part 1</a><a href="/paper/risk">Part 2</a>'
    return render(md, title=title, nav=nav)


def figure(name: str) -> Path | None:
    base = (DOCS / "figures").resolve()
    target = (base / name).resolve()
    if base not in target.parents or not target.is_file():
        return None
    return target
