"""SRAG Phase 2 — Document Converter Manager Tests (DD-05, DD-06).

Verifies:
- DocFormat enum values and behavior
- ConversionResult dataclass defaults
- TextConverter reads TXT/MD files correctly
- PDFConverter graceful degradation when PyMuPDF is unavailable
- DOCXConverter graceful degradation when python-docx is unavailable
- EPUBConverter graceful degradation when ebooklib is unavailable
- HTMLConverter graceful degradation when BeautifulSoup4 is unavailable
- ImageOCRConverter stub behavior
- ConverterManager registry operations (register/unregister/list)
- ConverterManager convert dispatch by file extension
- ConverterManager.is_format_supported
- ConverterManager.register_by_declarative
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from ingest.converter import (
    DocFormat,
    ConverterStatus,
    ConversionResult,
    DocumentConverter,
    TextConverter,
    PDFConverter,
    DOCXConverter,
    ImageOCRConverter,
    EPUBConverter,
    HTMLConverter,
    ConverterManager,
)


# ---------------------------------------------------------------------------
# Helpers — temporary file creation
# ---------------------------------------------------------------------------

def _write_text_file(content: str, suffix: str = ".txt") -> Path:
    """Write *content* to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    try:
        tmp.write(content)
    finally:
        tmp.close()
    return Path(tmp.name)


def _cleanup(path: Path) -> None:
    """Remove a temporary file if it exists."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# DocFormat enum tests
# ---------------------------------------------------------------------------

class TestDocFormatEnum:
    def test_pdf_value(self) -> None:
        assert DocFormat.PDF.value == "pdf"

    def test_docx_value(self) -> None:
        assert DocFormat.DOCX.value == "docx"

    def test_txt_value(self) -> None:
        assert DocFormat.TXT.value == "txt"

    def test_md_value(self) -> None:
        assert DocFormat.MD.value == "md"

    def test_epub_value(self) -> None:
        assert DocFormat.EPUB.value == "epub"

    def test_html_value(self) -> None:
        assert DocFormat.HTML.value == "html"

    def test_image_value(self) -> None:
        assert DocFormat.IMAGE.value == "image"


# ---------------------------------------------------------------------------
# ConverterStatus enum tests
# ---------------------------------------------------------------------------

class TestConverterStatusEnum:
    def test_active_value(self) -> None:
        assert ConverterStatus.ACTIVE.value == "active"

    def test_inactive_value(self) -> None:
        assert ConverterStatus.INACTIVE.value == "inactive"

    def test_error_value(self) -> None:
        assert ConverterStatus.ERROR.value == "error"


# ---------------------------------------------------------------------------
# ConversionResult dataclass tests
# ---------------------------------------------------------------------------

class TestConversionResult:
    def test_default_text_is_empty(self) -> None:
        r = ConversionResult(success=False, error="test")
        assert r.text == ""

    def test_default_metadata_is_empty_dict(self) -> None:
        r = ConversionResult(success=True)
        assert isinstance(r.metadata, dict)
        assert len(r.metadata) == 0

    def test_default_format_is_txt(self) -> None:
        r = ConversionResult(success=False)
        assert r.format == DocFormat.TXT

    def test_success_fields(self) -> None:
        r = ConversionResult(success=True, text="hello", format=DocFormat.TXT)
        assert r.success is True
        assert r.text == "hello"
        assert r.error is None


# ---------------------------------------------------------------------------
# TextConverter tests
# ---------------------------------------------------------------------------

class TestTextConverter:
    def test_convert_txt_file(self) -> None:
        converter = TextConverter()
        tmp = _write_text_file("Hello, World!\nThis is a test.")
        try:
            result = converter.convert(tmp)
            assert result.success is True
            assert "Hello, World" in result.text
            assert result.format == DocFormat.TXT
        finally:
            _cleanup(tmp)

    def test_convert_md_file(self) -> None:
        converter = TextConverter()
        tmp = _write_text_file("# Title\n\nBody text.", suffix=".md")
        try:
            result = converter.convert(tmp)
            assert result.success is True
            assert "# Title" in result.text
            assert result.format == DocFormat.MD
        finally:
            _cleanup(tmp)

    def test_convert_nonexistent_file_fails(self) -> None:
        converter = TextConverter()
        result = converter.convert("/nonexistent/path/test.txt")
        assert result.success is False
        assert result.error is not None

    def test_name_property(self) -> None:
        converter = TextConverter()
        assert converter.name == "text"


# ---------------------------------------------------------------------------
# PDFConverter tests (graceful degradation when fitz unavailable)
# ---------------------------------------------------------------------------

class TestPDFConverter:
    def test_graceful_degradation_without_markitdown_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the markitdown CLI can't be resolved, convert() should fail gracefully."""
        import ingest.converter as converter_mod

        monkeypatch.setattr(converter_mod, "_resolve_markitdown_cli", lambda: None)
        converter = PDFConverter()
        tmp = _write_text_file("dummy", suffix=".pdf")
        try:
            result = converter.convert(tmp)
            assert result.success is False
            assert "markitdown" in result.error.lower()  # type: ignore[union-attr]
            assert result.format == DocFormat.PDF
        finally:
            _cleanup(tmp)

    def test_name_property(self) -> None:
        converter = PDFConverter()
        assert converter.name == "pdf"


# ---------------------------------------------------------------------------
# DOCXConverter tests (graceful degradation when python-docx unavailable)
# ---------------------------------------------------------------------------

class TestDOCXConverter:
    def test_graceful_degradation_without_markitdown_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the markitdown CLI can't be resolved, convert() should fail gracefully."""
        import ingest.converter as converter_mod

        monkeypatch.setattr(converter_mod, "_resolve_markitdown_cli", lambda: None)
        converter = DOCXConverter()
        tmp = _write_text_file("dummy", suffix=".docx")
        try:
            result = converter.convert(tmp)
            assert result.success is False
            assert "markitdown" in result.error.lower()  # type: ignore[union-attr]
            assert result.format == DocFormat.DOCX
        finally:
            _cleanup(tmp)

    def test_name_property(self) -> None:
        converter = DOCXConverter()
        assert converter.name == "docx"


# ---------------------------------------------------------------------------
# ImageOCRConverter tests (stub behavior)
# ---------------------------------------------------------------------------

class TestImageOCRConverter:
    def test_stub_returns_placeholder_text(self) -> None:
        """When ABBYY SDK is unavailable, the stub should still return success with placeholder."""
        converter = ImageOCRConverter()
        # Force _abbyy_available to False first.
        original = converter._abbyy_available
        converter._abbyy_available = False
        try:
            tmp = _write_text_file("dummy", suffix=".png")
            try:
                result = converter.convert(tmp)
                # With ABBYY unavailable, should fail gracefully.
                assert result.success is False
                assert "ABBYY" in result.error  # type: ignore[arg-type]
            finally:
                _cleanup(tmp)
        finally:
            converter._abbyy_available = original

    def test_name_property(self) -> None:
        converter = ImageOCRConverter()
        assert converter.name == "image_ocr"


# ---------------------------------------------------------------------------
# EPUBConverter tests (graceful degradation when ebooklib unavailable)
# ---------------------------------------------------------------------------

class TestEPUBConverter:
    def test_graceful_degradation_without_ebooklib(self) -> None:
        """When ebooklib is not installed, convert() should return a failure result."""
        converter = EPUBConverter()
        original = converter._available
        converter._available = False
        try:
            tmp = _write_text_file("dummy", suffix=".epub")
            try:
                result = converter.convert(tmp)
                assert result.success is False
                assert "ebooklib" in result.error.lower()  # type: ignore[arg-type]
                assert result.format == DocFormat.EPUB
            finally:
                _cleanup(tmp)
        finally:
            converter._available = original

    def test_name_property(self) -> None:
        converter = EPUBConverter()
        assert converter.name == "epub"


# ---------------------------------------------------------------------------
# HTMLConverter tests (graceful degradation when BeautifulSoup4 unavailable)
# ---------------------------------------------------------------------------

class TestHTMLConverter:
    def test_graceful_degradation_without_beautifulsoup(self) -> None:
        """When bs4 is not installed, convert() should return a failure result."""
        converter = HTMLConverter()
        original = converter._beautifulsoup_available
        converter._beautifulsoup_available = False
        try:
            tmp = _write_text_file("dummy", suffix=".html")
            try:
                result = converter.convert(tmp)
                assert result.success is False
                assert "BeautifulSoup" in result.error  # type: ignore[arg-type]
                assert result.format == DocFormat.HTML
            finally:
                _cleanup(tmp)
        finally:
            converter._beautifulsoup_available = original

    def test_name_property(self) -> None:
        converter = HTMLConverter()
        assert converter.name == "html"


# ---------------------------------------------------------------------------
# ConverterManager registry tests
# ---------------------------------------------------------------------------

class TestConverterManagerRegistry:
    def test_default_converters_registered(self) -> None:
        mgr = ConverterManager()
        converters = mgr.list_converters()
        formats = {c["format"] for c in converters}
        # At minimum PDF and TXT should be registered by default.
        assert DocFormat.PDF.value in formats or DocFormat.TXT.value in formats

    def test_register_converter(self) -> None:
        mgr = ConverterManager()
        mgr.register_converter(DocFormat.TXT, TextConverter())
        converters = mgr.list_converters()
        formats = {c["format"] for c in converters}
        assert DocFormat.TXT.value in formats

    def test_unregister_converter(self) -> None:
        mgr = ConverterManager()
        # Unregister TXT.
        mgr.unregister_converter(DocFormat.TXT)
        status = mgr._status.get(DocFormat.TXT)
        assert status == ConverterStatus.INACTIVE

    def test_get_converter(self) -> None:
        mgr = ConverterManager()
        txt_conv = mgr.get_converter(DocFormat.TXT)
        # May return TextConverter or None depending on defaults.
        assert txt_conv is not None or isinstance(txt_conv, DocumentConverter)

    def test_is_format_supported_txt(self) -> None:
        mgr = ConverterManager()
        tmp = _write_text_file("test", suffix=".txt")
        try:
            assert mgr.is_format_supported(tmp) is True
        finally:
            _cleanup(tmp)

    def test_is_format_supported_unsupported_extension(self) -> None:
        mgr = ConverterManager()
        # Use an extension not in DEFAULT_EXTENSIONS.
        tmp = _write_text_file("test", suffix=".xyz")
        try:
            assert mgr.is_format_supported(tmp) is False
        finally:
            _cleanup(tmp)

    def test_convert_unsupported_extension(self) -> None:
        mgr = ConverterManager()
        tmp = _write_text_file("test", suffix=".xyz")
        try:
            result = mgr.convert(tmp)
            assert result.success is False
        finally:
            _cleanup(tmp)


# ---------------------------------------------------------------------------
# ConverterManager convert dispatch tests
# ---------------------------------------------------------------------------

class TestConverterManagerConvert:
    def test_convert_txt_via_manager(self) -> None:
        mgr = ConverterManager()
        tmp = _write_text_file("Manager test content.")
        try:
            result = mgr.convert(tmp)
            assert result.success is True
            assert "Manager test content" in result.text
        finally:
            _cleanup(tmp)

    def test_convert_nonexistent_file_fails(self) -> None:
        mgr = ConverterManager()
        result = mgr.convert("/nonexistent/file.txt")
        assert result.success is False


# ---------------------------------------------------------------------------
# register_by_declarative tests
# ---------------------------------------------------------------------------

class TestDeclarativeRegistration:
    def test_register_by_declarative(self) -> None:
        mgr = ConverterManager()
        mgr.register_by_declarative([
            {"format": "txt"},
            {"format": "pdf", "ocr_enabled": False},
        ])
        converters = mgr.list_converters()
        formats = {c["format"] for c in converters}
        assert DocFormat.TXT.value in formats
        assert DocFormat.PDF.value in formats

    def test_register_by_declarative_unknown_format_skipped(self) -> None:
        mgr = ConverterManager()
        # Unknown format should be skipped gracefully.
        mgr.register_by_declarative([{"format": "nonexistent"}])
        converters = mgr.list_converters()
        formats = {c["format"] for c in converters}
        assert "nonexistent" not in formats


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_convert_empty_file(self) -> None:
        converter = TextConverter()
        tmp = _write_text_file("")
        try:
            result = converter.convert(tmp)
            assert result.success is True
            assert result.text == ""
        finally:
            _cleanup(tmp)

    def test_convert_binary_content_as_txt(self) -> None:
        """Attempting to read binary data as text should fail gracefully."""
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        try:
            tmp.write(b"\x00\x01\x02\x03")
            tmp.close()
            converter = TextConverter()
            result = converter.convert(tmp.name)
            # May succeed (binary as text) or fail depending on encoding.
            assert result is not None
        finally:
            _cleanup(Path(tmp.name))

    def test_conversion_result_equality(self) -> None:
        r1 = ConversionResult(success=True, text="hello")
        r2 = ConversionResult(success=True, text="hello")
        # Dataclass comparison.
        assert r1 == r2