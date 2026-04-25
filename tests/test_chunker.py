"""Unit tests for the chunker."""

from src.parsing.chunker import chunk

def test_chunker_blocks_below_target_are_merged_into_one_chunk() -> None:
    text = (
        "Senior software engineer with eight years of backend experience.\n\n"
        "Education: BSc in Computer Science from a reputable university.")

    result = chunk(text, target_chars=400)

    assert len(result) == 1
    assert "Senior software engineer" in result[0]
    assert "Education:" in result[0]


def test_chunker_block_above_target_emits_immediately() -> None:
    long_block = "x" * 500
    text = f"{long_block}\n\nshort tail block that is long enough to survive the filter."

    result = chunk(text, target_chars=400)

    assert len(result) == 2
    assert result[0] == long_block
    assert result[1].startswith("short tail block")


def test_chunker_short_block_below_min_chars_is_dropped_before_merge() -> None:
    text = (
        "Page 1\n\n"
        "Detailed paragraph that comfortably exceeds the default minimum length.")

    result = chunk(text)

    assert len(result) == 1
    assert result[0].startswith("Detailed paragraph")
    assert "Page 1" not in result[0]


def test_chunker_whitespace_only_blocks_are_dropped() -> None:
    text = (
        "Real content block that is long enough to survive the filter step.\n\n"
        "   \n\n"
        "Another real content block which also clears the minimum threshold.")

    result = chunk(text, target_chars=10000)

    assert len(result) == 1
    assert "Real content block" in result[0]
    assert "Another real content block" in result[0]


def test_chunker_empty_string_returns_empty_list() -> None:
    text = ""
    result = chunk(text)
    assert result == []


def test_chunker_trailing_buffer_below_target_is_still_emitted() -> None:
    long_block = "y" * 500
    tail = "tail block long enough to clear the minimum length filter for chunks."
    text = f"{long_block}\n\n{tail}"

    result = chunk(text, target_chars=400)

    assert len(result) == 2
    assert result[1] == tail


def test_chunker_merged_blocks_are_separated_by_double_newline() -> None:
    block_a = "First content block long enough to clear the minimum filter threshold."
    block_b = "Second content block also long enough to survive minimum filtering."
    text = f"{block_a}\n\n{block_b}"

    result = chunk(text, target_chars=10000)

    assert len(result) == 1
    assert result[0] == f"{block_a}\n\n{block_b}"
