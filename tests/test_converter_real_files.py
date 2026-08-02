"""Real-file DOCX/PDF conversion tests (FD-11, FD-12).

Unlike tests/test_converter.py (which mostly exercises graceful degradation
when optional libraries are missing), these tests generate real, valid PDF
and DOCX files on disk and run them through the actual markitdown-CLI
conversion path end to end, asserting the extracted text is correct. This
also guards against the specific reversed-text bug the markitdown-CLI
routing exists to avoid (see ingest/converter.py's module docstring).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ingest.converter import ConverterManager, DocFormat, DOCXConverter, PDFConverter

PDF_MARKER = "UNIQUETESTMARKERPDF98765"
DOCX_MARKER = "UNIQUETESTMARKERDOCX43210"

pytestmark = pytest.mark.skipif(
    __import__("ingest.converter", fromlist=["_resolve_markitdown_cli"])._resolve_markitdown_cli() is None,
    reason="markitdown CLI not available in this environment",
)


def _make_real_pdf(path: Path, text: str) -> None:
    """Write a real, valid single-page PDF containing *text*."""
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.drawString(72, 720, text)
    c.save()


def _make_real_docx(path: Path, paragraphs: list[str]) -> None:
    """Write a real, minimal-but-valid OOXML .docx containing *paragraphs*.

    Hand-built via zipfile (no python-docx dependency in this project) —
    the smallest structure Word/markitdown will reliably parse.
    """
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    body_paragraphs = "".join(
        f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_paragraphs}</w:body>"
        "</w:document>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)


class TestPDFConverterRealFile:
    def test_converts_real_pdf_and_extracts_text(self, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        _make_real_pdf(pdf_path, PDF_MARKER)

        result = PDFConverter().convert(pdf_path)

        assert result.success is True, result.error
        assert result.format == DocFormat.PDF
        assert PDF_MARKER in result.text

    def test_extracted_text_is_not_reversed(self, tmp_path):
        """Regression guard for the exact bug the CLI routing exists to avoid."""
        pdf_path = tmp_path / "sample.pdf"
        _make_real_pdf(pdf_path, "the quick brown fox")
        result = PDFConverter().convert(pdf_path)

        assert "the" in result.text
        assert "eht" not in result.text  # "the" reversed

    def test_nonexistent_pdf_fails_gracefully(self, tmp_path):
        result = PDFConverter().convert(tmp_path / "does_not_exist.pdf")
        assert result.success is False
        assert result.error


class TestDOCXConverterRealFile:
    def test_converts_real_docx_and_extracts_text(self, tmp_path):
        docx_path = tmp_path / "sample.docx"
        _make_real_docx(docx_path, [DOCX_MARKER, "Second paragraph of real content."])

        result = DOCXConverter().convert(docx_path)

        assert result.success is True, result.error
        assert result.format == DocFormat.DOCX
        assert DOCX_MARKER in result.text
        assert "Second paragraph" in result.text

    def test_nonexistent_docx_fails_gracefully(self, tmp_path):
        result = DOCXConverter().convert(tmp_path / "does_not_exist.docx")
        assert result.success is False
        assert result.error


class TestConverterManagerRealFiles:
    def test_manager_routes_pdf_by_extension(self, tmp_path):
        pdf_path = tmp_path / "report.pdf"
        _make_real_pdf(pdf_path, PDF_MARKER)

        result = ConverterManager().convert(str(pdf_path))

        assert result.success is True, result.error
        assert PDF_MARKER in result.text

    def test_manager_routes_docx_by_extension(self, tmp_path):
        docx_path = tmp_path / "report.docx"
        _make_real_docx(docx_path, [DOCX_MARKER])

        result = ConverterManager().convert(str(docx_path))

        assert result.success is True, result.error
        assert DOCX_MARKER in result.text
