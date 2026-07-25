"""Render reports/REPORT.md (with figures) into a styled PDF."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from PIL import Image
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD_PATH = REPORTS / "REPORT.md"
PDF_PATH = REPORTS / "REPORT.pdf"

# A4 content box after margins, in points
PAGE_WIDTH_PT = 595.28
MARGIN_PT = 48
CONTENT_WIDTH_PT = PAGE_WIDTH_PT - 2 * MARGIN_PT
MAX_IMAGE_HEIGHT_PT = 340

CSS = """
@page {
    size: a4 portrait;
    margin: 48pt 48pt 56pt 48pt;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 20pt; left: 48pt; right: 48pt; height: 20pt;
    }
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22pt; color: #10243e; margin: 0 0 6pt 0; line-height: 1.25; }
h2 { font-size: 15pt; color: #10243e; margin: 20pt 0 6pt 0; border-bottom: 1pt solid #d0d7de; padding-bottom: 3pt; }
h3 { font-size: 12pt; color: #24405e; margin: 14pt 0 4pt 0; }
p { margin: 0 0 8pt 0; text-align: justify; }
em { color: #555; }
hr { border: 0; border-top: 1pt solid #d0d7de; margin: 14pt 0; }
a { color: #0b62c4; text-decoration: none; }
ul, ol { margin: 0 0 8pt 14pt; }
li { margin-bottom: 4pt; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 12pt 0; font-size: 9pt; }
th { background-color: #10243e; color: #ffffff; padding: 5pt; text-align: left; font-weight: bold; }
td { padding: 5pt; border-bottom: 0.5pt solid #d0d7de; }
tr:nth-child(even) td { background-color: #f4f6f8; }
img { margin: 6pt 0; }
.figure { -pdf-keep-with-next: true; margin: 8pt 0 4pt 0; }
.caption { font-size: 8.5pt; color: #666; font-style: italic; text-align: center; margin: 0 0 12pt 0; }
.cover-sub { font-size: 12pt; color: #4a5c70; font-style: italic; margin-bottom: 10pt; }
.meta { font-size: 9pt; color: #666; }
#footerContent { text-align: center; font-size: 8pt; color: #999; }
"""


def sized_figure(src: Path, caption: str | None) -> str:
    """Scale image to fit the content box, keeping its caption in the same block."""
    with Image.open(src) as im:
        w_px, h_px = im.size
    aspect = w_px / h_px
    width = min(CONTENT_WIDTH_PT, MAX_IMAGE_HEIGHT_PT * aspect)
    height = width / aspect
    caption_html = f'<p class="caption">{caption}</p>' if caption else ""
    return (
        f'<div class="figure"><img src="{src.as_posix()}" '
        f'width="{width:.0f}" height="{height:.0f}" />{caption_html}</div>'
    )


def build_html() -> str:
    md_text = MD_PATH.read_text(encoding="utf-8")

    # Strip the leading H1/subtitle: they are rendered as a styled cover block.
    lines = md_text.split("\n")
    title = lines[0].lstrip("# ").strip()
    body_md = "\n".join(lines[1:]).lstrip("\n")
    subtitle = ""
    if body_md.startswith("*"):
        first_para, _, rest = body_md.partition("\n\n")
        subtitle = first_para.strip().strip("*")
        body_md = rest

    html_body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "sane_lists"])

    # An image paragraph, optionally followed by an all-italic paragraph, becomes
    # a single figure block so the caption can never be split from its image.
    def replace_figure(match: re.Match) -> str:
        src = match.group("src")
        img_path = (REPORTS / src).resolve()
        if not img_path.exists():
            print(f"  WARNING: missing image {src}", file=sys.stderr)
            return ""
        return sized_figure(img_path, match.group("caption"))

    html_body = re.sub(
        r'<p>\s*<img[^>]*src="(?P<src>[^"]+)"[^>]*/?>\s*</p>'
        r"(?:\s*<p><em>(?P<caption>.*?)</em></p>)?",
        replace_figure,
        html_body,
        flags=re.DOTALL,
    )

    cover = f"""
    <h1>{title}</h1>
    <p class="cover-sub">{subtitle}</p>
    <p class="meta">Adey Innovations Inc. &middot; Fraud Detection for E-commerce and Bank Transactions</p>
    <hr />
    """

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8" /><style>{CSS}</style></head>
<body>
<div id="footerContent">Adey Innovations Inc. — Fraud Detection Report — page <pdf:pagenumber />
</div>
{cover}
{html_body}
</body></html>"""


def main() -> int:
    html = build_html()
    (REPORTS / "REPORT.html").write_text(html, encoding="utf-8")

    with open(PDF_PATH, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")

    if result.err:
        print(f"FAILED with {result.err} error(s)", file=sys.stderr)
        return 1
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
