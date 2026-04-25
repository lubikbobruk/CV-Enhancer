"""Unit tests for PDF and DOCX text extraction."""

import io
from pathlib import Path
import pytest
from src.parsing.pdf_parser import extract_text as pdf_extract_text
from src.parsing.docx_parser import extract_text as docx_extract_text

DUMMY_DOCS_DIR = Path(__file__).resolve().parent / "dummy_docs"

def _open(name: str):
    return open(DUMMY_DOCS_DIR / name, "rb")


# ------------------------------------------------
# .pdf format parsing unit testing
# ------------------------------------------------


class TestPdfSimple:

    def test_pdf_simple_contains_name(self):
        with _open("01_simple.pdf") as f:
            text = pdf_extract_text(f)
        assert "John Smith" in text

    def test_pdf_simple_contains_content(self):
        with _open("01_simple.pdf") as f:
            text = pdf_extract_text(f)
        assert "5 years" in text
        assert "Python" in text
        assert "TechCorp" in text

    def test_pdf_simple_returns_nonempty_string(self):
        with _open("01_simple.pdf") as f:
            text = pdf_extract_text(f)
        assert isinstance(text, str)
        assert len(text) > 50


class TestPdfMultisection:

    def test_pdf_multisection_contains_all_sections(self):
        with _open("02_multisection.pdf") as f:
            text = pdf_extract_text(f)
        assert "Summary" in text
        assert "Experience" in text
        assert "Education" in text

    def test_pdf_multisection_preserves_order(self):
        with _open("02_multisection.pdf") as f:
            text = pdf_extract_text(f)
        assert text.index("Summary") < text.index("Experience")
        assert text.index("Experience") < text.index("Education")

    def test_pdf_multisection_contains_details(self):
        with _open("02_multisection.pdf") as f:
            text = pdf_extract_text(f)
        assert "React" in text
        assert "StartupCo" in text
        assert "Czech Technical University" in text


class TestPdfSkills:
    def test_pdf_skills_contains_technologies(self):
        with _open("03_skills.pdf") as f:
            text = pdf_extract_text(f)
        for skill in ["Python", "JavaScript", "Kubernetes", "Docker", "AWS"]:
            assert skill in text, f"Missing skill: {skill}"


class TestPdfShortLines:
    def test_pdf_short_lines_contains_real_content(self):
        with _open("04_short_lines.pdf") as f:
            text = pdf_extract_text(f)
        assert "Senior Data Scientist" in text
        assert "XGBoost" in text

    def test_pdf_short_lines_preserves_short_fragments(self):
        with _open("04_short_lines.pdf") as f:
            text = pdf_extract_text(f)
        # Short lines should still be present in raw extraction
        assert "Maria Garcia" in text


class TestPdfMinimal:
    def test_pdf_minimal_extracts_something(self):
        with _open("05_minimal.pdf") as f:
            text = pdf_extract_text(f)
        assert "Tom Brown" in text

    def test_pdf_minimal_is_short(self):
        with _open("05_minimal.pdf") as f:
            text = pdf_extract_text(f)
        assert len(text) < 100


class TestPdfMultipage:
    def test_pdf_multipage_contains_all_roles(self):
        with _open("06_multipage.pdf") as f:
            text = pdf_extract_text(f)
        for company in ["BigTech", "MidCorp", "StartupX", "ConsultCo"]:
            assert company in text, f"Missing company: {company}"

    def test_pdf_multipage_contains_education(self):
        with _open("06_multipage.pdf") as f:
            text = pdf_extract_text(f)
        assert "Stanford" in text
        assert "Cambridge" in text

    def test_pdf_multipage_has_blank_line_between_pages(self):
        with _open("06_multipage.pdf") as f:
            text = pdf_extract_text(f)
        # Must have at least one \n\n 
        assert "\n\n" in text


class TestPdfTable:
    def test_pdf_table_extracts_table_content(self):
        with _open("07_table.pdf") as f:
            text = pdf_extract_text(f)
        assert "Python" in text
        assert "Expert" in text

    def test_pdf_table_extracts_non_table_content(self):
        with _open("07_table.pdf") as f:
            text = pdf_extract_text(f)
        assert "Backend Engineer" in text
        assert "FinTech" in text


class TestPdfUnicode:
    def test_pdf_unicode_preserves_czech_characters(self):
        with _open("08_unicode.pdf") as f:
            text = pdf_extract_text(f)
        assert "Tomáš" in text or "Tomas" in text
        assert "Novák" in text or "Novak" in text

    def test_pdf_unicode_contains_czech_content(self):
        with _open("08_unicode.pdf") as f:
            text = pdf_extract_text(f)
        assert "Radiokomunikace" in text or "inžen" in text


class TestPdfBullets:
    def test_pdf_bullets_contains_all_bullet_content(self):
        with _open("09_bullets.pdf") as f:
            text = pdf_extract_text(f)
        assert "transformer" in text
        assert "ONNX" in text
        assert "BERT" in text

    def test_pdf_bullets_preserves_company_info(self):
        with _open("09_bullets.pdf") as f:
            text = pdf_extract_text(f)
        assert "AIStartup" in text
        assert "ResearchLab" in text


class TestPdfRepeated:
    def test_pdf_repeated_preserves_all_mentions(self):
        with _open("10_repeated.pdf") as f:
            text = pdf_extract_text(f)
        # "Python" appears many times in this fixture
        assert text.count("Python") >= 5


# ------------------------------------------------
# .docx format parsing unit testing
# ------------------------------------------------


class TestDocxSimple:
    def test_docx_simple_contains_name(self):
        with _open("01_simple.docx") as f:
            text = docx_extract_text(f)
        assert "John Smith" in text

    def test_docx_simple_has_blank_line_separator(self):
        with _open("01_simple.docx") as f:
            text = docx_extract_text(f)
        # Two paragraphs should be separated by \n\n
        assert "\n\n" in text

    def test_docx_simple_contains_content(self):
        with _open("01_simple.docx") as f:
            text = docx_extract_text(f)
        assert "Python" in text
        assert "TechCorp" in text


class TestDocxMultisection:
    def test_docx_multisection_preserves_order(self):
        with _open("02_multisection.docx") as f:
            text = docx_extract_text(f)
        assert text.index("Summary") < text.index("Experience")
        assert text.index("Experience") < text.index("Education")

    def test_docx_multisection_separates_sections(self):
        with _open("02_multisection.docx") as f:
            text = docx_extract_text(f)
        # Must have multiple \n\n separators between sections
        assert text.count("\n\n") >= 3


class TestDocxSkills:
    def test_docx_skills_contains_all_technologies(self):
        with _open("03_skills.docx") as f:
            text = docx_extract_text(f)
        for skill in ["Python", "JavaScript", "Kubernetes", "Docker", "AWS", "Terraform"]:
            assert skill in text, f"Missing skill: {skill}"


class TestDocxShortLines:
    def test_docx_short_lines_preserves_everything(self):
        with _open("04_short_lines.docx") as f:
            text = docx_extract_text(f)
        assert "Maria Garcia" in text
        assert "Prague, CZ" in text
        assert "2024" in text
        assert "XGBoost" in text


class TestDocxMinimal:
    def test_docx_minimal_extracts_name(self):
        with _open("05_minimal.docx") as f:
            text = docx_extract_text(f)
        assert "Tom Brown" in text

    def test_docx_minimal_no_blank_lines(self):
        with _open("05_minimal.docx") as f:
            text = docx_extract_text(f)
        assert "\n\n" not in text


class TestDocxMultipage:
    def test_docx_multipage_contains_all_companies(self):
        with _open("06_multipage.docx") as f:
            text = docx_extract_text(f)
        for company in ["BigTech", "MidCorp", "StartupX", "ConsultCo"]:
            assert company in text, f"Missing company: {company}"

    def test_docx_multipage_preserves_publications(self):
        with _open("06_multipage.docx") as f:
            text = docx_extract_text(f)
        assert "VLDB" in text
        assert "Byzantine" in text

    def test_docx_multipage_has_many_separators(self):
        with _open("06_multipage.docx") as f:
            text = docx_extract_text(f)
        assert text.count("\n\n") >= 5


class TestDocxTable:
    def test_docx_table_extracts_table_as_pipes(self):
        with _open("07_table.docx") as f:
            text = docx_extract_text(f)
        # Table cells should be joined with "|"
        assert "|" in text

    def test_docx_table_contains_table_values(self):
        with _open("07_table.docx") as f:
            text = docx_extract_text(f)
        assert "Python" in text
        assert "Expert" in text
        assert "JavaScript" in text

    def test_docx_table_preserves_non_table_content(self):
        with _open("07_table.docx") as f:
            text = docx_extract_text(f)
        assert "Backend Engineer" in text
        assert "FinTech" in text


class TestDocxUnicode:
    def test_docx_unicode_preserves_diacritics(self):
        with _open("08_unicode.docx") as f:
            text = docx_extract_text(f)
        assert "Tomáš" in text
        assert "Novák" in text

    def test_docx_unicode_preserves_czech_content(self):
        with _open("08_unicode.docx") as f:
            text = docx_extract_text(f)
        assert "inženýr" in text
        assert "Radiokomunikace" in text


class TestDocxBullets:
    def test_docx_bullets_contains_all_items(self):
        with _open("09_bullets.docx") as f:
            text = docx_extract_text(f)
        assert "transformer" in text
        assert "ONNX" in text
        assert "BERT" in text

    def test_docx_bullets_preserves_both_roles(self):
        with _open("09_bullets.docx") as f:
            text = docx_extract_text(f)
        assert "AIStartup" in text
        assert "ResearchLab" in text


class TestDocxRepeated:
    def test_docx_repeated_preserves_all_mentions(self):
        with _open("10_repeated.docx") as f:
            text = docx_extract_text(f)
        assert text.count("Python") >= 5