"""Parse dispatcher. Import aggregator."""

from pathlib import Path
from src.parsing import docx_parser, pdf_parser
from src.parsing.chunker import chunk

# Local branch
_PARSERS = {
    ".pdf": pdf_parser.extract_text,
    ".docx": docx_parser.extract_text,
}

def extract_text(stream, filename) -> str:
    """Dispatch to the right parser based on the filename extension."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")
    return _PARSERS[suffix](stream)

# What to import on from src.parsing import *
__all__ = ["extract_text", "chunk"]