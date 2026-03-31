#!/usr/bin/env python3
"""
md_to_pdf.py — Convert a Markdown file to a styled PDF.

Usage:
    uv run --with markdown --with weasyprint tools/md_to_pdf.py INPUT.md
    uv run --with markdown --with weasyprint tools/md_to_pdf.py INPUT.md OUT.pdf
"""

import argparse
import os
import re
import sys
from pathlib import Path

# On Windows, WeasyPrint needs GTK DLLs on PATH / in the DLL search path.
# Probe common install locations and add the first one found.
if sys.platform == "win32":
    _GTK_CANDIDATES = [
        r"C:\Program Files\GTK3-Runtime Win64\bin",
        r"C:\Program Files\GTK4-Runtime Win64\bin",
        r"C:\gtk\bin",
        r"C:\gtk3\bin",
        r"C:\msys64\mingw64\bin",
        r"C:\msys64\ucrt64\bin",
    ]
    for _gtk_bin in _GTK_CANDIDATES:
        if Path(_gtk_bin).is_dir():
            os.add_dll_directory(_gtk_bin)
            os.environ.setdefault(
                "PATH", os.environ.get("PATH", "") + os.pathsep + _gtk_bin
            )
            os.environ["PATH"] = (
                os.environ["PATH"] + os.pathsep + _gtk_bin
            )
            break

import markdown
from weasyprint import HTML

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@font-face {
    font-family: 'FallbackSans';
    src: local('Segoe UI'), local('Inter'), local('Helvetica Neue'),
         local('Arial');
}

@page {
    size: A4;
    margin: 2.8cm 3cm 2.8cm 3cm;
}

@page :first {
    margin-top: 4cm;
}

@page {
    @bottom-center {
        content: counter(page);
        font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
        font-size: 8.5pt;
        color: #999;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
}

/* ---- Title block ---- */

h1 {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 38pt;
    font-weight: 800;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 6pt;
    color: #0d0d0d;
    margin-top: 0;
    margin-bottom: 0.15em;
    border: none;
    line-height: 1.1;
}

/* Subtitle + author lines immediately after h1 */
h1 + p,
h1 + p + p {
    text-align: center;
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 10.5pt;
    color: #555;
    margin-top: 0.1em;
    margin-bottom: 0.2em;
    line-height: 1.5;
}

/* Horizontal rule separating title block from body */
h1 ~ hr:first-of-type {
    margin-top: 1.8em;
    margin-bottom: 0;
    border: none;
    border-top: 2px solid #222;
}

/* ---- Section headings ---- */

h2 {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 15pt;
    font-weight: 700;
    color: #111;
    margin-top: 0;
    margin-bottom: 0.5em;
    padding-bottom: 0.25em;
    border-bottom: 2px solid #222;
    line-height: 1.2;
    break-before: page;
    page-break-before: always;
    break-after: avoid;
    page-break-after: avoid;
}

h2.no-break {
    break-before: auto;
    page-break-before: auto;
}

h3 {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 12pt;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 1.6em;
    margin-bottom: 0.35em;
    line-height: 1.3;
    break-after: avoid;
    page-break-after: avoid;
}

h4 {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 10.5pt;
    font-weight: 600;
    font-style: italic;
    color: #333;
    margin-top: 1.3em;
    margin-bottom: 0.3em;
    line-height: 1.3;
    break-after: avoid;
    page-break-after: avoid;
}

/* ---- Body text ---- */

p {
    margin-top: 0;
    margin-bottom: 0.8em;
    orphans: 3;
    widows: 3;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.8em 0;
}

/* ---- Lists ---- */

ul, ol {
    padding-left: 1.6em;
    margin-top: 0.3em;
    margin-bottom: 0.8em;
}

li {
    margin-bottom: 0.3em;
    line-height: 1.65;
}

li > p {
    margin-bottom: 0.2em;
}

/* ---- Blockquotes ---- */

blockquote {
    margin: 1.4em 0;
    padding: 0.75em 1.1em;
    border-left: 4px solid #444;
    background-color: #f5f5f5;
    color: #2a2a2a;
    font-style: italic;
    border-radius: 0 3px 3px 0;
    break-inside: avoid;
    page-break-inside: avoid;
}

blockquote p {
    margin: 0 0 0.3em 0;
    line-height: 1.6;
}

blockquote p:last-child {
    margin-bottom: 0;
}

blockquote strong {
    font-style: normal;
    color: #111;
}

/* ---- Inline ---- */

strong {
    font-weight: 700;
    color: #0d0d0d;
}

em {
    font-style: italic;
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background: #f0f0f0;
    padding: 0.1em 0.3em;
    border-radius: 2px;
}

pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 0.8em 1em;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    line-height: 1.5;
    overflow-x: auto;
    break-inside: avoid;
    page-break-inside: avoid;
}

/* ---- Tables ---- */

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 9.5pt;
    break-inside: avoid;
    page-break-inside: avoid;
}

th {
    background: #222;
    color: #fff;
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-weight: 600;
    padding: 0.4em 0.7em;
    text-align: left;
}

td {
    padding: 0.35em 0.7em;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) td {
    background: #f9f9f9;
}
"""

# ---------------------------------------------------------------------------
# Markdown extensions
# ---------------------------------------------------------------------------

MD_EXTENSIONS = [
    "extra",
    "sane_lists",
    "smarty",
    "toc",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def md_to_html_fragment(md_str: str) -> str:
    md = markdown.Markdown(extensions=MD_EXTENSIONS)
    return md.convert(md_str)


def suppress_first_h2_break(html: str) -> str:
    """Add class="no-break" to the first <h2> so it doesn't
    force a blank first page."""
    return re.sub(r"<h2>", '<h2 class="no-break">', html, count=1)


def build_full_html(fragment: str, title: str = "Document") -> str:
    fragment = suppress_first_h2_break(fragment)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{title}</title>\n"
        f"  <style>{CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{fragment}\n"
        "</body>\n"
        "</html>"
    )


def render_pdf(html_str: str, out_path: Path) -> None:
    HTML(string=html_str).write_pdf(str(out_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a Markdown file to a styled PDF."
    )
    p.add_argument("input", type=Path, help="Input .md file")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "Output .pdf file "
            "(default: same name as input, .pdf extension)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"Error: {args.input} does not exist")

    out = args.output or args.input.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    md_str = load_markdown(args.input)
    fragment = md_to_html_fragment(md_str)
    html = build_full_html(fragment, title=args.input.stem)
    render_pdf(html, out)

    print(f"Written: {out}")


if __name__ == "__main__":
    main()
