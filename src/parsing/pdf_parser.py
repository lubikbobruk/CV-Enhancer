"""PDF text + embedded-image extraction via pypdf."""

from pypdf import PdfReader


def extract_text(stream) -> str:
    """Extract text from a PDF binary stream. Pages joined by blank line."""
    stream.seek(0)
    reader = PdfReader(stream)
    pages = [page.extract_text(extraction_mode="layout").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def extract_largest_image(stream) -> bytes | None:
    """Return the raw bytes of the largest embedded image, or None if there are none."""
    stream.seek(0)
    reader = PdfReader(stream)

    best_area = 0
    best_bytes: bytes | None = None
    for page in reader.pages:
        for img in page.images:
            try:
                w, h = img.image.size
            except Exception:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best_bytes = img.data
    return best_bytes
