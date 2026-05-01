"""Split parsed CV/job-ad text into chunks for retrieval.

`chunk()` is the generic packer chunking job ad.
`chunk_cv()` is cv aware chunker.
"""

import re

def _build_role_header_re() -> "re.Pattern[str]":
    months = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December"
    dash   = r"\s*[-–—]\s*"

    numeric = r"(?:0?[1-9]|1[0-2])/\d{4}"   # 06/2025
    text    = rf"(?:{months})\s+\d{{4}}"    # June 2025
    year    = r"(?:19|20)\d{2}"             # 2025

    def with_optional_end(start: str) -> str:
        # "06/2025" alone, or "06/2025 – Present", or "06/2025 – 04/2026"
        return rf"{start}(?:{dash}(?:Present|{start}))?"

    shapes = "|".join([
        with_optional_end(numeric),
        with_optional_end(text),
        with_optional_end(year),
        "Present",
    ])
    return re.compile(rf"(?:{shapes})\s*$", re.IGNORECASE)


_role_header_re = _build_role_header_re()

_DEFAULT_ROLE_MAX_CHARS = 3000
_MIN_ROLES_TO_TRIGGER = 2


def chunk(text: str, min_chars: int = 40, target_chars: int = 400, max_chars: int = 600) -> list[str]:
    """Split on blank lines, drop tiny blocks, pack toward target_chars without exceeding max_chars."""
    units: list[str] = []
    for raw in text.split("\n\n"):
        block = raw.strip()
        if not block or len(block) < min_chars:
            continue
        if len(block) <= max_chars:
            units.append(block)
            continue
        for line in block.split("\n"):
            line = line.strip()
            if line and len(line) >= min_chars:
                units.append(line)

    # glue small units together
    result: list[str] = []
    buffer = ""
    for unit in units:
        if not buffer:
            buffer = unit
            if len(buffer) >= target_chars:
                result.append(buffer)
                buffer = ""
            continue
        if len(buffer) + 2 + len(unit) > max_chars:
            result.append(buffer)
            buffer = unit
        else:
            buffer = buffer + "\n\n" + unit
        if len(buffer) >= target_chars:
            result.append(buffer)
            buffer = ""
    if buffer:
        result.append(buffer)
    return result


def _is_role_header(line: str) -> bool:
    # searchers for a header using regex, returns None if failed to find
    return bool(_role_header_re.search(line.rstrip()))


def chunk_cv(text: str, role_max_chars: int = _DEFAULT_ROLE_MAX_CHARS) -> list[str]:
    """Chunk a CV by role boundaries; fall back to `chunk()` if too few roles detected."""
    lines = text.split("\n")
    role_indices = [i for i, line in enumerate(lines) if _is_role_header(line)]
    if len(role_indices) < _MIN_ROLES_TO_TRIGGER:
        return chunk(text)

    chunks: list[str] = []

    preamble = "\n".join(lines[:role_indices[0]]).strip()
    if preamble:
        chunks.extend(chunk(preamble))

    for idx, start in enumerate(role_indices):
        if idx + 1 < len(role_indices):
            end = role_indices[idx + 1]
            trailing_start = end
        # if last role section
        else:
            end = _find_section_end(lines, start)
            trailing_start = end

        role_text = "\n".join(lines[start:end]).strip()
        if role_text:
            if len(role_text) <= role_max_chars:
                chunks.append(role_text)
            else:
                chunks.extend(chunk(role_text, target_chars=role_max_chars // 2, max_chars=role_max_chars))

        if idx + 1 == len(role_indices):
            trailing = "\n".join(lines[trailing_start:]).strip()
            if trailing:
                chunks.extend(chunk(trailing))

    return chunks


def _find_section_end(lines: list[str], start: int) -> int:
    """Find the line index where a role's content ends — first run of >=2 blank lines."""
    blank_run = 0
    for i in range(start + 1, len(lines)):
        if not lines[i].strip():
            blank_run += 1
            if blank_run >= 2:
                return i - blank_run + 1
        else:
            blank_run = 0
    return len(lines)
