"""Split parsed CV/job-ad text into chunks for retrieval."""

def chunk(text: str, min_chars: int = 40, target_chars: int = 400) -> list[str]:
    """Split on blank lines, drop tiny blocks, merge short blocks to target the char size."""
    blocks = []
    for raw in text.split("\n\n"):
        block = raw.strip()
        if block and len(block) >= min_chars:
            blocks.append(block)

    result = []
    buffer = ""
    for block in blocks:
        if buffer:
            buffer = buffer + "\n\n" + block
        else:
            buffer = block
        if len(buffer) >= target_chars:
            result.append(buffer)
            buffer = ""
    if buffer:
        result.append(buffer)
    return result
