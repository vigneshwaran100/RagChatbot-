"""Unit tests for the document parser module."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.ingestion.parser import parse_document, _txt_to_markdown, _normalize_extension
from app.utils.exceptions import DocumentParseError, UnsupportedFileTypeError


class TestNormalizeExtension:
    def test_pdf(self):
        assert _normalize_extension("report.pdf") == "pdf"

    def test_markdown(self):
        assert _normalize_extension("notes.MD") == "md"

    def test_txt(self):
        assert _normalize_extension("data.TXT") == "txt"

    def test_no_extension(self):
        assert _normalize_extension("noext") == ""


class TestTxtToMarkdown:
    def test_adds_header(self):
        result = _txt_to_markdown("Hello world.", "my_doc.txt")
        assert result.startswith("# my_doc")

    def test_collapses_blank_lines(self):
        result = _txt_to_markdown("Para1.\n\n\n\nPara2.", "doc.txt")
        assert "\n\n\n" not in result


class TestParseDocument:
    def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedFileTypeError):
            parse_document(b"content", "file.docx")

    def test_txt_returns_markdown(self):
        content = b"This is plain text content about AI."
        result = parse_document(content, "test.txt")
        assert "test" in result
        assert "AI" in result

    def test_md_returns_content_unchanged(self):
        md = b"# Header\n\nSome markdown content."
        result = parse_document(md, "notes.md")
        assert "# Header" in result
        assert "markdown content" in result

    def test_empty_txt_handled(self):
        result = parse_document(b"   ", "empty.txt")
        # Should not raise; returns header at minimum
        assert isinstance(result, str)

    def test_pdf_calls_docling(self):
        with patch("app.ingestion.parser._parse_pdf") as mock_pdf:
            mock_pdf.return_value = "# Parsed PDF\n\nContent here."
            result = parse_document(b"%PDF-1.4 fake", "report.pdf")
            mock_pdf.assert_called_once()
            assert "Parsed PDF" in result
