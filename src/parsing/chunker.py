"""Split parsed CV/job-ad text into chunks for retrieval."""

def chunk(text: str,min_chars: int = 40,target_chars: int = 400,max_chars: int = 600) -> list[str]:
    """Split on blank lines, drop tiny blocks, pack toward target_chars without exceeding max_chars."""
    units: list[str] = []
    for raw in text.split("\n\n"):
        block = raw.strip()
        if not block or len(block) < min_chars:
            continue
        if len(block) <= max_chars:
            units.append(block)
            continue
        # Oversized: split on single newlines
        for line in block.split("\n"):
            line = line.strip()
            if line and len(line) >= min_chars:
                units.append(line)

    result: list[str] = []
    buffer = ""
    for unit in units:
        if not buffer:
            buffer = unit
            if len(buffer) >= target_chars:
                result.append(buffer)
                buffer = ""
            continue
        # Emit before appending if the join would breach max_chars
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
