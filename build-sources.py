#!/usr/bin/env python3
"""Render README.md to sources.html so the source list is readable on the live site.

Netlify serves .md as text/markdown, which browsers download rather than display —
so a footer link straight to README.md hides the very thing it points at. This
converts the subset of Markdown the README actually uses (headings, tables, fenced
code, blockquotes, lists, inline code/links/emphasis) into a page styled with the
site's own stylesheet.

Re-run after editing README.md:   python3 build-sources.py
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def inline(text):
    """Inline formatting. Code spans are stashed first so their contents survive."""
    stash = []

    def keep(m):
        stash.append(m.group(1))
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<a href="{html.escape(m.group(2), quote=True)}"'
                            f'{" target=\"_blank\" rel=\"noopener\"" if m.group(2).startswith("http") else ""}>'
                            f'{m.group(1)}</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: f"<code>{html.escape(stash[int(m.group(1))], quote=False)}</code>", text)


def convert(md):
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):                                  # fenced code
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i], quote=False))
                i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out.append('<div class="table-wrap"><table><thead><tr>'
                       + "".join(f"<th>{inline(c)}</th>" for c in cells)
                       + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table></div>")
            continue

        if m := re.match(r"^(#{1,6})\s+(.*)$", line):               # heading
            lvl = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f'<h{lvl} id="{slug}">{inline(m.group(2))}</h{lvl}>')
            i += 1
            continue

        if line.startswith(">"):                                    # blockquote
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>")
            continue

        if re.match(r"^[-*]\s+", line):                             # list (with continuations)
            out.append("<ul>")
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i]):
                item = [re.sub(r"^[-*]\s+", "", lines[i])]
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    item.append(lines[i].strip())
                    i += 1
                out.append(f"<li>{inline(' '.join(item))}</li>")
            out.append("</ul>")
            continue

        if not line.strip():
            i += 1
            continue

        buf = []                                                    # paragraph
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|```|[-*]\s)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            out.append(f"<p>{inline(' '.join(buf))}</p>")
    return "\n".join(out)


FAVICON = ("data:image/svg+xml,"
           "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Crect width='32' height='32' rx='7' fill='%230b0d12'/%3E"
           "%3Cpath d='M6 22V10m0 0l10 12M16 10v12m0-12l10 12M26 10v12' "
           "stroke='%236ea8ff' stroke-width='2.4' fill='none' stroke-linecap='round'/%3E%3C/svg%3E")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Sources — How Attention Actually Got Here</title>
<meta name="description" content="Every date in the attention timeline, with the primary source it was checked against and the method used to verify it." />
<link rel="icon" href="{favicon}" />
<link rel="stylesheet" href="styles.css" />
<style>
  .doc {{ max-width: 860px; margin: 0 auto; padding: 48px 24px 120px; }}
  .doc h1 {{ font-size: 2rem; letter-spacing: -0.01em; margin: 0 0 8px; }}
  .doc h2 {{ font-size: 1.35rem; margin: 44px 0 12px; padding-top: 20px; border-top: 1px solid var(--border); }}
  .doc h3 {{ font-size: 1.05rem; margin: 28px 0 10px; color: var(--text); }}
  .doc p, .doc li {{ color: var(--text-dim); font-size: 15px; }}
  .doc li {{ margin-bottom: 8px; }}
  .doc code {{ font-family: var(--mono); font-size: 0.9em; background: var(--bg-elevated);
               border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; color: var(--accent); }}
  .doc pre {{ background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px;
              padding: 14px 16px; overflow-x: auto; }}
  .doc pre code {{ background: none; border: none; padding: 0; color: var(--text-dim); font-size: 12.5px; line-height: 1.7; }}
  .doc blockquote {{ margin: 14px 0; padding: 10px 18px; border-left: 3px solid var(--accent-2);
                     background: var(--bg-elevated); color: var(--text-dim); font-size: 14.5px; }}
  .table-wrap {{ overflow-x: auto; margin: 16px 0; }}
  .doc table {{ border-collapse: collapse; width: 100%; min-width: 640px; font-size: 13px; }}
  .doc th {{ text-align: left; font-family: var(--mono); font-size: 10.5px; text-transform: uppercase;
             letter-spacing: 0.06em; color: var(--text-faint); border-bottom: 1px solid var(--border-strong);
             padding: 0 12px 8px 0; vertical-align: bottom; }}
  .doc td {{ padding: 10px 12px 10px 0; border-bottom: 1px solid var(--border);
             color: var(--text-dim); vertical-align: top; }}
  .backlink {{ font-family: var(--mono); font-size: 12.5px; display: inline-block; margin-bottom: 28px; }}
</style>
</head>
<body>
<div class="doc">
<a class="backlink" href="./">&#8592; back to the timeline</a>
{body}
</div>
</body>
</html>
"""


def main():
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    out = ROOT / "sources.html"
    out.write_text(PAGE.format(body=convert(md), favicon=FAVICON), encoding="utf-8")
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes) from README.md")


if __name__ == "__main__":
    sys.exit(main())
