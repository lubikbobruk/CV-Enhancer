"""Render CV chunks to a PDF byte stream via reportlab."""

from __future__ import annotations

import os
import re
from io import BytesIO

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.parsing.chunker import _role_header_re


_FONT_BODY = "CVBody"
_FONT_BOLD = "CVBodyBold"
_FONTS_REGISTERED = False

_ACCENT = colors.HexColor("#2c3e50")
_RULE = colors.HexColor("#bdc3c7")
_MUTED = colors.HexColor("#5d6d7e")
_TEXT = colors.HexColor("#212529")

_SECTION_LABELS = {
    "SUMMARY", "EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS",
    "INTERNATIONAL ACADEMIC EXCHANGES", "LANGUAGES", "CERTIFICATIONS",
    "AWARDS", "PUBLICATIONS", "VOLUNTEERING", "INTERESTS",
}

_BULLET_PREFIX_RE = re.compile(r"^\s*([■•◦▪●·*\-–])\s+")
_NAME_LINE_RE = re.compile(r"^[A-ZÀ-ÖØ-Ý][\w'`\-]*(?:\s+[A-ZÀ-ÖØ-Ý][\w'`\-]*){0,3}$")
_CONTACT_HINT_RE = re.compile(r"[@·∙•|]|\+\d|\bhttps?://|\.com\b", re.IGNORECASE)
_CONTACT_LINE_RE = re.compile(
    r"@|\+\d|\bhttps?://|\.com\b|\.org\b|\.io\b|\bLanguages?:|\b\d{3}\b"
)


# (regular, bold) — first set whose regular file exists wins.
_FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
)


def _register_fonts() -> None:
    """Register a wide-Unicode TTF family. Falls back to Bitstream Vera so Polish/Czech glyphs render."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return

    chosen = next((p for p in _FONT_CANDIDATES if os.path.isfile(p[0])), None)
    if chosen is None:
        fonts_dir = os.path.join(os.path.dirname(reportlab.__file__), "fonts")
        chosen = (os.path.join(fonts_dir, "Vera.ttf"), os.path.join(fonts_dir, "VeraBd.ttf"))

    regular, bold = chosen
    pdfmetrics.registerFont(TTFont(_FONT_BODY, regular))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD, bold if os.path.isfile(bold) else regular))
    _FONTS_REGISTERED = True


def _styles() -> dict:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        name="Body", parent=base["BodyText"], fontName=_FONT_BODY,
        fontSize=10, leading=13.5, spaceAfter=2, textColor=_TEXT, alignment=TA_LEFT,
    )
    bold = ParagraphStyle(name="Bold", parent=body, fontName=_FONT_BOLD)
    return {
        "body": body,
        "bullet": ParagraphStyle(
            name="Bullet", parent=body, leftIndent=14, firstLineIndent=-14, spaceAfter=1,
        ),
        "section": ParagraphStyle(
            name="Section", parent=bold, fontSize=12.5, leading=15,
            textColor=_ACCENT, spaceBefore=12, spaceAfter=2,
        ),
        "name": ParagraphStyle(
            name="Name", parent=bold, fontSize=22, leading=26,
            textColor=_ACCENT, spaceAfter=2,
        ),
        "tagline": ParagraphStyle(
            name="Tagline", parent=body, fontSize=11, leading=14,
            textColor=_MUTED, spaceAfter=4,
        ),
        "role_left": ParagraphStyle(
            name="RoleLeft", parent=bold, fontSize=11, leading=14,
            textColor=_ACCENT, spaceBefore=6, spaceAfter=0,
        ),
        "role_right": ParagraphStyle(
            name="RoleRight", parent=bold, fontSize=10, leading=14,
            textColor=_MUTED, alignment=TA_RIGHT, spaceBefore=6, spaceAfter=0,
        ),
    }


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _is_section_label(line: str) -> bool:
    return line.strip().rstrip(":").upper() in _SECTION_LABELS


def _split_role_header(line: str) -> tuple[str, str] | None:
    """If `line` ends with a date pattern, return (title, date)."""
    stripped = line.rstrip()
    m = _role_header_re.search(stripped)
    if not m:
        return None
    title = stripped[: m.start()].rstrip(" \t-–—·•|")
    if not title:
        return None
    return title, stripped[m.start():].strip()


def _is_contact_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and bool(_CONTACT_LINE_RE.search(s))


def _normalize_bullets(line: str) -> str:
    """Repair PDFs where the bullet glyph was lost in extraction (PUA / replacement char)."""
    stripped = line.lstrip()
    if not stripped:
        return line
    first = stripped[0]
    if first == "�" or "" <= first <= "":
        rest = stripped[1:].lstrip()
        if rest:
            return "■ " + rest
    return line


def _promote_indent_bullets(lines: list[str]) -> list[str]:
    """If a chunk uses bullets, promote leading-space lines to bullets too.

    pypdf sometimes drops a font-glyph bullet entirely, leaving just whitespace.
    Without this, those lines would silently merge into the previous paragraph.
    """
    if not any(_BULLET_PREFIX_RE.match(line) for line in lines):
        return lines
    out: list[str] = []
    for line in lines:
        if line.strip() and not _BULLET_PREFIX_RE.match(line) and line[:1] in (" ", "\t"):
            out.append("■ " + line.strip())
        else:
            out.append(line)
    return out


def _is_structural_break(line: str) -> bool:
    """Lines that should NOT be reflowed into adjacent ones."""
    s = line.strip()
    if not s:
        return True
    return _is_section_label(s) or _split_role_header(s) is not None or bool(_BULLET_PREFIX_RE.match(s))


def _reflow_lines(lines: list[str], *, in_header: bool = False) -> list[str]:
    """Merge soft-wrapped lines back into paragraphs.

    pypdf's layout extraction preserves visual line breaks, so a single sentence
    spans multiple `\\n`s. In header mode, contact/profile lines are also kept
    standalone so name/email/links don't collapse into one paragraph.
    """
    pre = _promote_indent_bullets([_normalize_bullets(line) for line in lines])

    out: list[str] = []
    for line in pre:
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        if not (prev.strip() and line.strip()):
            out.append(line)
            continue
        if _is_structural_break(prev) or _is_structural_break(line):
            out.append(line)
            continue
        if in_header and (_is_contact_line(prev) or _is_contact_line(line)):
            out.append(line)
            continue
        joiner = "" if prev.endswith("-") else " "
        out[-1] = prev.rstrip() + joiner + line.lstrip()
    return out


def _render_section(raw: str, styles: dict) -> list:
    return [
        Spacer(1, 0.15 * cm),
        Paragraph(_xml_escape(raw.strip().rstrip(":").upper()), styles["section"]),
        HRFlowable(width="100%", thickness=0.6, color=_RULE, spaceBefore=1, spaceAfter=4),
    ]


def _render_role_header(title: str, date: str, styles: dict) -> list:
    tbl = Table(
        [[
            Paragraph(_xml_escape(title), styles["role_left"]),
            Paragraph(_xml_escape(date), styles["role_right"]),
        ]],
        colWidths=[None, 4 * cm],
    )
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [tbl]


def _render_bullet(body: str, styles: dict) -> list:
    return [Paragraph(
        f"<font color='#2c3e50'>•</font>&nbsp;&nbsp;{_xml_escape(body)}",
        styles["bullet"],
    )]


def _render_line(line: str, styles: dict) -> list:
    raw = line.rstrip()
    if not raw:
        return [Spacer(1, 0.15 * cm)]
    if _is_section_label(raw):
        return _render_section(raw, styles)
    role = _split_role_header(raw)
    if role is not None:
        return _render_role_header(*role, styles)
    bullet_match = _BULLET_PREFIX_RE.match(raw)
    if bullet_match:
        return _render_bullet(raw[bullet_match.end():].strip(), styles)
    return [Paragraph(_xml_escape(raw), styles["body"])]


def _render_header_block(lines: list[str], styles: dict) -> tuple[list, int]:
    """Render the name + contact preamble. Returns (flowables, lines_consumed)."""
    i = next((k for k, line in enumerate(lines) if line.strip()), len(lines))
    if i >= len(lines):
        return [], 0

    first = lines[i].strip()
    if not _NAME_LINE_RE.match(first) or _CONTACT_HINT_RE.search(first):
        return [], 0

    flow: list = [Paragraph(_xml_escape(first), styles["name"])]

    captured = 0
    j = i + 1
    while j < len(lines) and captured < 6:
        line = lines[j].strip()
        if not line:
            j += 1
            continue
        if _is_section_label(line) or _split_role_header(line) is not None:
            break
        flow.append(Paragraph(_xml_escape(line), styles["tagline"]))
        captured += 1
        j += 1

    flow.append(HRFlowable(
        width="100%", thickness=0.8, color=_RULE, spaceBefore=4, spaceAfter=6,
    ))
    return flow, j


def build_pdf(chunks: list[str], overrides: dict[int, str] | None = None) -> bytes:
    """Render chunks (with overrides applied) to PDF bytes."""
    _register_fonts()
    overrides = overrides or {}
    styles = _styles()

    # Flatten chunks into one line stream so the header detector spans chunk
    # boundaries. Reflow per chunk to undo pypdf's visual line wrapping.
    all_lines: list[str] = []
    chunk_breaks: list[int] = []
    for idx, original in enumerate(chunks):
        text = overrides.get(idx, original)
        chunk_breaks.append(len(all_lines))
        all_lines.extend(_reflow_lines(text.split("\n"), in_header=(idx == 0)))

    flow: list = []
    header_flow, consumed = _render_header_block(all_lines, styles)
    flow.extend(header_flow)

    pending_breaks = [b for b in chunk_breaks if b > consumed]
    for i in range(consumed, len(all_lines)):
        if pending_breaks and i == pending_breaks[0]:
            flow.append(Spacer(1, 0.2 * cm))
            pending_breaks.pop(0)
        flow.extend(_render_line(all_lines[i], styles))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="Enhanced CV",
    )
    doc.build(flow)
    return buf.getvalue()
