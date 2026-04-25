"""DOCX text extraction iterating XML paragraphs/tables."""

from docx import Document
from docx.oxml.ns import qn

def extract_text(stream) -> str:
    """Extract text from a DOCX binary stream. Paragraphs and tables in order."""
    # move streamlits cursor back
    stream.seek(0)
    doc = Document(stream)
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        # paragraph w:p
        if child.tag == qn("w:p"):
            # iterate text of the paragraph
            text = "".join(node.text or "" for node in child.iter(qn("w:t"))).strip()
            if text:
                blocks.append(text)
        # table w:tbl
        elif child.tag == qn("w:tbl"):
            rows = []
            for row in child.iter(qn("w:tr")):
                # iterate text of the table rows/cols
                cells = [
                    "".join(node.text or "" for node in cell.iter(qn("w:t"))).strip()
                    for cell in row.iter(qn("w:tc"))
                ]
                rows.append(" | ".join(c for c in cells if c))
            table_text = "\n".join(r for r in rows   if r)
            if table_text:
                blocks.append(table_text)
    return "\n\n".join(blocks)
