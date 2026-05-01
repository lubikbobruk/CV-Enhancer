"""Unit tests for the chunker."""

from src.parsing.chunker import chunk, chunk_cv

def test_chunker_short_blocks_merge() -> None:
    text = (
        "Senior software engineer with eight years of backend experience.\n\n"
        "Education: BSc in Computer Science from a reputable university.")

    result = chunk(text, target_chars=400)

    assert len(result) == 1
    assert "Senior software engineer" in result[0]
    assert "Education:" in result[0]


def test_chunker_oversized_block_emits_alone() -> None:
    long_block = "x" * 500
    text = f"{long_block}\n\nshort tail block that is long enough to survive the filter."

    result = chunk(text, target_chars=400, max_chars=500)

    assert len(result) == 2
    assert result[0] == long_block
    assert result[1].startswith("short tail block")


def test_chunker_tiny_block_dropped() -> None:
    text = (
        "Page 1\n\n"
        "Detailed paragraph that comfortably exceeds the default minimum length.")

    result = chunk(text)

    assert len(result) == 1
    assert result[0].startswith("Detailed paragraph")
    assert "Page 1" not in result[0]


def test_chunker_whitespace_block_dropped() -> None:
    text = (
        "Real content block that is long enough to survive the filter step.\n\n"
        "   \n\n"
        "Another real content block which also clears the minimum threshold.")

    result = chunk(text, target_chars=10000)

    assert len(result) == 1
    assert "Real content block" in result[0]
    assert "Another real content block" in result[0]


def test_chunker_empty_input_returns_empty() -> None:
    text = ""
    result = chunk(text)
    assert result == []


def test_chunker_trailing_buffer_emitted() -> None:
    long_block = "y" * 500
    tail = "tail block long enough to clear the minimum length filter for chunks."
    text = f"{long_block}\n\n{tail}"

    result = chunk(text, target_chars=400, max_chars=500)

    assert len(result) == 2
    assert result[1] == tail


def test_chunker_merge_uses_blank_line_separator() -> None:
    block_a = "First content block long enough to clear the minimum filter threshold."
    block_b = "Second content block also long enough to survive minimum filtering."
    text = f"{block_a}\n\n{block_b}"

    result = chunk(text, target_chars=10000)

    assert len(result) == 1
    assert result[0] == f"{block_a}\n\n{block_b}"


def test_chunker_oversized_block_splits_on_newlines() -> None:
    line = "x" * 80
    big_block = "\n".join([line] * 10)  # 10 lines, ~810 chars total — one block.
    result = chunk(big_block, min_chars=40, target_chars=200, max_chars=300)

    assert len(result) > 1
    assert all(len(c) <= 300 for c in result)


def test_chunker_pack_respects_max_chars() -> None:
    block_a = "a" * 300
    block_b = "b" * 300
    text = f"{block_a}\n\n{block_b}"
    result = chunk(text, min_chars=40, target_chars=400, max_chars=500)

    assert len(result) == 2
    assert result[0] == block_a
    assert result[1] == block_b


def test_chunk_cv_one_chunk_per_role() -> None:
    text = (
        "Acme Corp — Backend Engineer    06/2025 – Present\n"
        "Built data pipelines and REST services.\n"
        "Stack: Python, PostgreSQL, Redis.\n"
        "\n"
        "Globex Inc — Software Developer    07/2023 – 04/2025\n"
        "Shipped a logistics dashboard used by warehouse staff.\n"
        "Wrote the full TypeScript frontend.")

    result = chunk_cv(text)

    assert len(result) == 2
    assert "Acme Corp" in result[0]
    assert "Python, PostgreSQL" in result[0]
    assert "Globex Inc" in result[1]
    assert "TypeScript frontend" in result[1]


def test_chunk_cv_preamble_separated_from_role() -> None:
    preamble = (
        "Jane Doe — Backend Engineer\n"
        "Backend engineer shipping reliable services into production. "
        "Strong software-engineering foundation in distributed systems and pragmatic delivery.")
    text = (
        f"{preamble}\n\n"
        "Acme Corp — Backend Engineer    06/2025 – Present\n"
        "Built data pipelines and REST services.\n\n"
        "Globex Inc — Software Developer    07/2023 – 04/2025\n"
        "Shipped a logistics dashboard.")

    result = chunk_cv(text)

    assert any("pragmatic delivery" in c for c in result)
    assert any("Acme Corp" in c and "pragmatic delivery" not in c for c in result)



def test_chunk_cv_mid_sentence_year_ignored() -> None:
    text = (
        "Senior researcher with publications on distributed consensus.\n"
        "Wilson et al. 2023. Adaptive consensus for geo-distributed databases. "
        "Proceedings of VLDB.\n"
        "Wilson et al. 2021. Practical Byzantine fault tolerance at scale.")

    result = chunk_cv(text)

    assert result == chunk(text)


def test_chunk_cv_single_year_below_threshold_falls_back() -> None:
    text = (
        "Senior software engineer with deep distributed-systems expertise.\n\n"
        "Started building production services back in 2015\n\n"
        "Active contributor to open-source observability tooling and packaging ecosystems.")

    result = chunk_cv(text)

    assert result == chunk(text)


def test_chunk_cv_text_month_dates_detected() -> None:
    text = (
        "Acme Corp — Backend Engineer    Jun 2025 – Present\n"
        "Built data pipelines and REST services.\n"
        "Stack: Python, PostgreSQL, Redis.\n"
        "\n"
        "Globex Inc — Software Developer    August 2023 – April 2025\n"
        "Shipped a logistics dashboard used by warehouse staff.\n"
        "Wrote the full TypeScript frontend.")

    result = chunk_cv(text)

    assert len(result) == 2
    assert "Acme Corp" in result[0]
    assert "Python, PostgreSQL" in result[0]
    assert "Globex Inc" in result[1]
    assert "TypeScript frontend" in result[1]