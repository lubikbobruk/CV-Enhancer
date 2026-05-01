"""Unit tests for the job-ad role/benefits filter."""

import pytest

from src.parsing.job_ad_filter import filter_role_relevant

def test_filter_no_model_passthrough() -> None:
    ad = (
        "Why Join Us? Yoga lessons, 25 vacation days, cafeteria allowance.\n\n"
        "Your Role — GenAI Engineer building Python applications focused on GenAI.")

    result = filter_role_relevant(ad, model=None)
    assert result == ad


def test_filter_empty_input_passthrough() -> None:
    ad = ""
    result = filter_role_relevant(ad, model=None)
    assert result == ""


def test_filter_drops_benefits_keeps_role() -> None:
    pytest.importorskip("sentence_transformers")
    from src.retrieval.embedding_retriever import load_minilm

    ad = (
        "Why Join Us? Attractive Benefits Package: private health care, fully paid sick leave, "
        "25 vacation days regardless of age, monthly cafeteria allowance, fully paid meal vouchers, "
        "smartphone with 1TB data, well-being and yoga lessons, team running challenges.\n\n"
        "Your Role — GenAI Engineer. Support development of scalable Python applications focused "
        "on GenAI. Apply techniques like document chunking and embeddings. Required: 1-3 years of "
        "Python experience, interest in LLMs, experience with Docker and GitHub Actions.")
    model = load_minilm()

    result = filter_role_relevant(ad, model=model)

    assert "GenAI Engineer" in result
    assert "Python" in result
    assert "cafeteria" not in result.lower()
    assert "yoga" not in result.lower()


def test_filter_drops_contact_block() -> None:
    pytest.importorskip("sentence_transformers")
    from src.retrieval.embedding_retriever import load_minilm

    ad = (
        "Your Role — Senior Python Engineer building distributed data pipelines, deploying to "
        "Kubernetes, and integrating with PostgreSQL and Kafka.\n\n"
        "Contact: Jane Doe — jane.doe@example.com, +420 123 456 789. "
        "Connect on LinkedIn: linkedin.com/in/janedoe")
    model = load_minilm()

    result = filter_role_relevant(ad, model=model)

    assert "Python Engineer" in result
    assert "@" not in result
    assert "linkedin.com" not in result.lower()


def test_filter_all_dropped_returns_original() -> None:
    pytest.importorskip("sentence_transformers")
    from src.retrieval.embedding_retriever import load_minilm

    ad = (
        "Attractive Benefits Package: private health care, paid sick leave, 25 vacation days, "
        "monthly cafeteria allowance, meal vouchers, smartphone, laptop.\n\n"
        "Well-being: yoga lessons, mental health webinars, running challenges, team sports, "
        "volunteering days, mental health support.")
    model = load_minilm()
    result = filter_role_relevant(ad, model=model)

    assert result == ad
