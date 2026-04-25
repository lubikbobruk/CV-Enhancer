"""PDF text extraction via pypdf."""

from pypdf import PdfReader

def extract_text(stream) -> str:
    """Extract text from a PDF binary stream. Pages joined by blank line."""
    # move streamlits cursor back
    stream.seek(0)
    reader = PdfReader(stream)
    pages = [page.extract_text(extraction_mode="layout").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)
