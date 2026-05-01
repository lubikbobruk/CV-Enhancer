"""Render an ordered list of CV chunks to a PDF byte stream via reportlab.

`build_pdf(chunks, overrides)` returns PDF bytes:
- chunks: original chunk list (in order).
- overrides: chunk_id -> replacement text (used for accepted Gemini rewrites).

Role-header lines (matched by chunker._role_header_re) render in bold.
Other lines render as normal paragraphs. Chunks separated by spacer.
"""

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.parsing.chunker import _role_header_re


def _styles() -> dict:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        name="Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=2,
    )
    bold = ParagraphStyle(
        name="Bold",
        parent=body,
        fontName="Helvetica-Bold",
    )
    return {"body": body, "bold": bold}


def _line_to_paragraph(line: str, styles: dict) -> Paragraph:
    style = styles["bold"] if _role_header_re.search(line.rstrip()) else styles["body"]
    safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe or "&nbsp;", style)


def build_pdf(chunks: list[str], overrides: dict[int, str] | None = None) -> bytes:
    """Render chunks (with overrides applied) to PDF bytes."""
    overrides = overrides or {}
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    flow: list = []
    for idx, original in enumerate(chunks):
        text = overrides.get(idx, original)
        for line in text.split("\n"):
            flow.append(_line_to_paragraph(line, styles))
        flow.append(Spacer(1, 0.4 * cm))

    doc.build(flow)
    return buf.getvalue()
