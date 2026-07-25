"""
pdf_export.py — converts markdown (a notebook export, or a single answer)
into a downloadable, styled PDF. Uses `markdown` + `xhtml2pdf`, both pure
Python with no system-level dependencies (unlike weasyprint), so it installs
cleanly on Windows with just pip.
"""

import io
import markdown as md_lib
from xhtml2pdf import pisa

PDF_TEMPLATE = """
<html>
<head>
<style>
  body {{ font-family: Helvetica, sans-serif; color: #1F1F1F; font-size: 11pt; line-height: 1.5; }}
  h1 {{ font-size: 20pt; color: #1F1F1F; margin-bottom: 4px; }}
  h2 {{ font-size: 14pt; color: #0B57D0; margin-top: 20px; margin-bottom: 8px; }}
  ul {{ margin: 6px 0; padding-left: 18px; }}
  li {{ margin-bottom: 6px; }}
  em {{ color: #5F6368; font-size: 9pt; }}
  .meta {{ color: #5F6368; font-size: 9pt; margin-bottom: 20px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_pdf_bytes(markdown_text: str, title: str = "AI Librarian Export") -> bytes:
    html_body = md_lib.markdown(markdown_text, extensions=["extra"])
    full_html = PDF_TEMPLATE.format(body=html_body)

    buffer = io.BytesIO()
    pisa.CreatePDF(src=full_html, dest=buffer)
    return buffer.getvalue()
