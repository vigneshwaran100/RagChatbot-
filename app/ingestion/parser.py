from __future__ import annotations

import re
import tempfile
from pathlib import Path

from app.utils.exceptions import DocumentParseError, UnsupportedFileTypeError
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "txt", "md"}


def _normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()


def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Parse a PDF, TXT, or Markdown file into a clean Markdown string
    using Docling for PDF and native reading for text-based formats.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect extension).

    Returns:
        Markdown-formatted string of the document content.

    Raises:
        UnsupportedFileTypeError: For unrecognised file extensions.
        DocumentParseError:       When parsing fails.
    """
    ext = _normalize_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}'. "
            f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Parsing document '%s' (type=%s, size=%d bytes)", filename, ext, len(file_bytes))

    try:
        if ext == "pdf":
            return _parse_pdf(file_bytes, filename)
        else:
            # TXT and MD are read as UTF-8; MD is already Markdown
            text = file_bytes.decode("utf-8", errors="replace")
            if ext == "txt":
                # Wrap plain text in minimal Markdown structure
                text = _txt_to_markdown(text, filename)
            return text
    except (UnsupportedFileTypeError, DocumentParseError):
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"Failed to parse '{filename}': {exc}"
        ) from exc


def _parse_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Use Docling to convert a PDF to Markdown.
    Writes bytes to a temp file because Docling works with file paths.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise DocumentParseError(
            "Docling is not installed. Run: pip install docling"
        ) from exc

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        converter = DocumentConverter()
        result = converter.convert(str(tmp_path))
        markdown = result.document.export_to_markdown()
        logger.debug(
            "Docling parsed '%s' → %d chars of Markdown", filename, len(markdown)
        )
        return markdown
    finally:
        tmp_path.unlink(missing_ok=True)


def _txt_to_markdown(text: str, filename: str) -> str:
    """
    Convert plain-text content to minimal Markdown by adding a header
    and normalising paragraph breaks.
    """
    stem = Path(filename).stem
    header = f"# {stem}\n\n"
    # Normalise multiple blank lines to double newline
    body = re.sub(r"\n{3,}", "\n\n", text.strip())
    return header + body
