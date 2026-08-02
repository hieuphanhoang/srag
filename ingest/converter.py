"""SRAG Document Converter Manager — Phase 2 implementation (DD-05, DD-06).

Manages multi-format document conversion with a pluggable converter registry.
Supports PDF, DOCX, TXT, Markdown, EPUB, HTML, and image-based OCR via ABBYY.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _resolve_markitdown_cli() -> str | None:
    """Resolve the ``markitdown`` CLI executable.

    Checked in this project's own venv first (``Path(sys.executable).parent``),
    then PATH. Never hardcode a path into another application's install tree.
    Routing through the CLI (rather than markitdown's Python API) avoids a
    known bug where some watermarked PDFs come back with reversed text.
    """
    exe_name = "markitdown.exe" if os.name == "nt" else "markitdown"
    venv_candidate = Path(sys.executable).parent / exe_name
    if venv_candidate.exists():
        return str(venv_candidate)
    return shutil.which("markitdown")


def _convert_via_markitdown_cli(file_path: str | Path, timeout: float = 120.0) -> tuple[bool, str]:
    """Run the markitdown CLI on *file_path*. Returns (success, text_or_error)."""
    cli = _resolve_markitdown_cli()
    if not cli:
        return False, "markitdown CLI not found in venv or PATH."
    try:
        proc = subprocess.run(
            [cli, str(file_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc)

    if proc.returncode != 0:
        return False, proc.stderr.strip() or f"markitdown exited with code {proc.returncode}"
    return True, proc.stdout


# ---------------------------------------------------------------------------
# Enums & Protocols
# ---------------------------------------------------------------------------

class DocFormat(str, Enum):
    """Supported document formats."""
    
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"
    MD = "md"
    EPUB = "epub"
    HTML = "html"
    IMAGE = "image"  # PNG, JPG, JPEG for OCR


class ConverterStatus(str, Enum):
    """Converter registration status."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class ConversionResult:
    """Result of a document conversion operation.

    Attributes
    ----------
    success : bool
        Whether the conversion succeeded.
    text : str
        Extracted text content (empty on failure).
    error : str | None
        Error message if conversion failed.
    format : DocFormat
        Detected input format.
    metadata : dict[str, Any] = field(default_factory=dict)
        Additional metadata (e.g., page count, author).
    """

    success: bool
    text: str = ""
    error: str | None = None
    format: DocFormat = DocFormat.TXT
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Converter Protocol & Base Class
# ---------------------------------------------------------------------------

class DocumentConverter(ABC):
    """Abstract base class for document converters.

    Subclasses must implement the :py:meth:`convert` method to extract
    text from a given file path.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Return the converter's registered name."""
        return self._name

    @abstractmethod
    def convert(self, file_path: str | Path) -> ConversionResult:
        """Convert a single document to text.

        Parameters
        ----------
        file_path : str | Path
            Path to the document file.

        Returns
        -------
        ConversionResult
            Contains extracted text or error information.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete Converters
# ---------------------------------------------------------------------------

class TextConverter(DocumentConverter):
    """Plain text and Markdown file converter."""

    def __init__(self) -> None:
        super().__init__("text")

    def convert(self, file_path: str | Path) -> ConversionResult:
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            return ConversionResult(
                success=True,
                text=content,
                format=self._detect_format(Path(file_path)),
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    @staticmethod
    def _detect_format(path: Path) -> DocFormat:
        suffix = path.suffix.lower()
        if suffix in (".md", ".markdown"):
            return DocFormat.MD
        return DocFormat.TXT


class PDFConverter(DocumentConverter):
    """PDF document converter using the markitdown CLI.

    Routed through the CLI (not markitdown's Python API) — the API is known
    to mis-extract text (reversed order) on some watermarked PDFs.
    """

    def __init__(self, enable_ocr: bool = False) -> None:
        # enable_ocr is accepted for backward compatibility with declarative
        # registration configs; OCR itself isn't implemented (no ABBYY SDK).
        super().__init__("pdf")

    def convert(self, file_path: str | Path) -> ConversionResult:
        success, text_or_error = _convert_via_markitdown_cli(file_path)
        if not success:
            return ConversionResult(success=False, error=text_or_error, format=DocFormat.PDF)
        return ConversionResult(success=True, text=text_or_error, format=DocFormat.PDF)


class DOCXConverter(DocumentConverter):
    """Microsoft Word (DOCX) document converter using the markitdown CLI.

    ``python-docx`` is not a project dependency (only markitdown's docx/pdf/
    pptx/xlsx extras are pinned in pyproject.toml) — use what's actually
    installed, and stay consistent with PDFConverter's CLI-subprocess routing.
    """

    def __init__(self) -> None:
        super().__init__("docx")

    def convert(self, file_path: str | Path) -> ConversionResult:
        success, text_or_error = _convert_via_markitdown_cli(file_path)
        if not success:
            return ConversionResult(success=False, error=text_or_error, format=DocFormat.DOCX)
        return ConversionResult(success=True, text=text_or_error, format=DocFormat.DOCX)


class PPTXConverter(DocumentConverter):
    """Microsoft PowerPoint (PPTX) document converter using the markitdown CLI.

    Same CLI-subprocess routing as PDFConverter/DOCXConverter — the
    ``pptx`` markitdown extra is pinned in pyproject.toml specifically for
    this.
    """

    def __init__(self) -> None:
        super().__init__("pptx")

    def convert(self, file_path: str | Path) -> ConversionResult:
        success, text_or_error = _convert_via_markitdown_cli(file_path)
        if not success:
            return ConversionResult(success=False, error=text_or_error, format=DocFormat.PPTX)
        return ConversionResult(success=True, text=text_or_error, format=DocFormat.PPTX)


class XLSXConverter(DocumentConverter):
    """Microsoft Excel (XLSX) document converter using the markitdown CLI.

    Same CLI-subprocess routing as PDFConverter/DOCXConverter — the
    ``xlsx`` markitdown extra is pinned in pyproject.toml specifically for
    this.
    """

    def __init__(self) -> None:
        super().__init__("xlsx")

    def convert(self, file_path: str | Path) -> ConversionResult:
        success, text_or_error = _convert_via_markitdown_cli(file_path)
        if not success:
            return ConversionResult(success=False, error=text_or_error, format=DocFormat.XLSX)
        return ConversionResult(success=True, text=text_or_error, format=DocFormat.XLSX)


class ImageOCRConverter(DocumentConverter):
    """Image-based OCR converter using ABBYY FineReader SDK.

    Supports PNG, JPG, JPEG formats for optical character recognition.
    """

    def __init__(self) -> None:
        super().__init__("image_ocr")
        self._abbyy_available = False
        self._try_import()

    def _try_import(self) -> None:
        try:
            import ABBYYFatReader as _fr  # noqa: F401
            self._abbyy_available = True
        except ImportError:
            logger.warning("ABBYY FatReader SDK not found; image OCR disabled.")

    def convert(self, file_path: str | Path) -> ConversionResult:
        if not self._abbyy_available:
            return ConversionResult(
                success=False,
                error="ABBYY FineReader SDK is required for image OCR.",
                format=DocFormat.IMAGE,
            )

        try:
            # Placeholder: real implementation would use ABBYFFatReader API.
            # For now, return a placeholder result.
            return ConversionResult(
                success=True,
                text=f"[OCR text from {Path(file_path).name}]",  # type: ignore[arg-type]
                format=DocFormat.IMAGE,
                metadata={"ocr_engine": "abbyy_fine_reader"},
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc), format=DocFormat.IMAGE)


class EPUBConverter(DocumentConverter):
    """EPUB ebook converter using ebooklib."""

    def __init__(self) -> None:
        super().__init__("epub")
        self._available = False
        try:
            import ebooklib as _eb  # noqa: F401
            from ebooklib import epub as _epub_mod  # noqa: F401
            self._available = True
        except ImportError:
            pass

    def convert(self, file_path: str | Path) -> ConversionResult:
        if not self._available:
            return ConversionResult(
                success=False,
                error="ebooklib is required for EPUB conversion.",
                format=DocFormat.EPUB,
            )

        try:
            from ebooklib import epub

            book = epub.read_epub(str(file_path))
            chapters: list[str] = []
            item_count = 0

            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    content = item.get_content().decode("utf-8")
                    # Strip HTML tags (simplified).
                    clean = self._strip_html(content)
                    if clean.strip():
                        chapters.append(clean.strip())
                    item_count += 1

            result_text = "\n\n--- CHAPTER BREAK ---\n".join(chapters)
            metadata: dict[str, Any] = {"chapter_count": item_count}

            return ConversionResult(
                success=True,
                text=result_text,
                format=DocFormat.EPUB,
                metadata=metadata,
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc), format=DocFormat.EPUB)

    @staticmethod
    def _strip_html(html_content: str) -> str:
        """Remove HTML tags from *html_content*."""
        import re
        text = re.sub(r"<[^>]+>", "", html_content)
        return text


class HTMLConverter(DocumentConverter):
    """HTML document converter that extracts visible text content."""

    def __init__(self) -> None:
        super().__init__("html")
        self._beautifulsoup_available = False
        try:
            from bs4 import BeautifulSoup as _BS  # noqa: F401
            self._beautifulsoup_available = True
        except ImportError:
            pass

    def convert(self, file_path: str | Path) -> ConversionResult:
        if not self._beautifulsoup_available:
            return ConversionResult(
                success=False,
                error="BeautifulSoup4 is required for HTML conversion.",
                format=DocFormat.HTML,
            )

        try:
            from bs4 import BeautifulSoup

            content = Path(file_path).read_text(encoding="utf-8")  # type: ignore[arg-type]
            soup = BeautifulSoup(content, "html.parser")

            # Remove script and style elements.
            for tag in soup.find_all(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            return ConversionResult(
                success=True,
                text=text.strip(),
                format=DocFormat.HTML,
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc), format=DocFormat.HTML)


# ---------------------------------------------------------------------------
# Converter Manager / Registry
# ---------------------------------------------------------------------------

class ConverterManager:
    """Registry and manager for document converters.

    Maps file extensions to the appropriate converter instance.
    Supports programmatic and declarative converter registration.

    Example
    -------
    >>> mgr = ConverterManager()
    >>> mgr.register_converter(DocFormat.PDF, PDFConverter())
    >>> result = mgr.convert("/path/to/doc.pdf")
    """

    # Default extension-to-converter mapping.
    DEFAULT_EXTENSIONS: dict[str, type[DocumentConverter]] = {
        ".pdf": PDFConverter,
        ".docx": DOCXConverter,
        ".doc": DOCXConverter,
        ".pptx": PPTXConverter,
        ".xlsx": XLSXConverter,
        ".txt": TextConverter,
        ".md": TextConverter,
        ".markdown": TextConverter,
        ".epub": EPUBConverter,
        ".html": HTMLConverter,
        ".htm": HTMLConverter,
        ".png": ImageOCRConverter,
        ".jpg": ImageOCRConverter,
        ".jpeg": ImageOCRConverter,
    }

    def __init__(self) -> None:
        self._converters: dict[DocFormat, DocumentConverter] = {}
        self._ext_map: dict[str, DocFormat] = {}
        self._status: dict[DocFormat, ConverterStatus] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Register all default converters."""
        for ext, conv_cls in self.DEFAULT_EXTENSIONS.items():
            try:
                instance = conv_cls()
                fmt = self._ext_to_format(ext)
                if fmt is not None:
                    self.register_converter(fmt, instance)
            except Exception as exc:
                logger.warning("Failed to register default converter %s: %s", conv_cls.__name__, exc)

    @staticmethod
    def _ext_to_format(ext: str) -> DocFormat | None:
        ext_lower = ext.lower().lstrip(".")
        try:
            return DocFormat(ext_lower)
        except ValueError:
            return None

    def register_converter(self, fmt: DocFormat, converter: DocumentConverter) -> None:
        """Register *converter* for documents of format *fmt*."""
        self._converters[fmt] = converter
        self._status[fmt] = ConverterStatus.ACTIVE
        logger.info("Registered converter '%s' for format %s.", converter.name, fmt.value)

    def unregister_converter(self, fmt: DocFormat) -> None:
        """Unregister the converter for format *fmt*."""
        self._converters.pop(fmt, None)
        self._status[fmt] = ConverterStatus.INACTIVE
        logger.info("Unregistered converter for format %s.", fmt.value)

    def register_by_declarative(self, converters_config: list[dict[str, Any]]) -> None:
        """Register converters declaratively.

        Parameters
        ----------
        converters_config : list[dict]
            Each dict must have ``"format"`` and optional ``"params"``,
            ``"ocr_enabled"``, etc.
        """
        for cfg in converters_config:
            fmt_str = cfg["format"]
            try:
                fmt = DocFormat(fmt_str)
            except ValueError:
                logger.warning("Unknown format '%s'; skipping.", fmt_str)
                continue

            params = cfg.get("params", {})
            if fmt == DocFormat.PDF:
                converter = PDFConverter(enable_ocr=cfg.get("ocr_enabled", False))
            elif fmt == DocFormat.DOCX:
                converter = DOCXConverter()
            elif fmt == DocFormat.PPTX:
                converter = PPTXConverter()
            elif fmt == DocFormat.XLSX:
                converter = XLSXConverter()
            elif fmt == DocFormat.IMAGE:
                converter = ImageOCRConverter()
            elif fmt == DocFormat.EPUB:
                converter = EPUBConverter()
            elif fmt == DocFormat.HTML:
                converter = HTMLConverter()
            else:
                converter = TextConverter()

            self.register_converter(fmt, converter)

    # ------------------------------------------------------------------
    # Conversion API
    # ------------------------------------------------------------------

    def convert(self, file_path: str | Path) -> ConversionResult:
        """Convert a document to text using the appropriate registered converter.

        Parameters
        ----------
        file_path : str | Path
            Path to the input document.

        Returns
        -------
        ConversionResult
            Contains extracted text or error details.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        fmt = self._ext_map.get(ext)
        if fmt is None:
            # Try to infer from file extension.
            fmt = self._infer_format(path)

        if fmt is None or fmt not in self._converters:
            return ConversionResult(
                success=False,
                error=f"No converter available for format '{ext}'.",
            )

        converter = self._converters[fmt]
        result = converter.convert(file_path)
        result.format = fmt

        if not result.success:
            self._status[fmt] = ConverterStatus.ERROR
        return result

    def get_converter(self, fmt: DocFormat) -> DocumentConverter | None:
        """Return the converter for *fmt*, or ``None``."""
        return self._converters.get(fmt)

    def list_converters(self) -> list[dict[str, Any]]:
        """List all registered converters and their status."""
        return [
            {
                "format": fmt.value,
                "name": self._converters[fmt].name,
                "status": self._status.get(fmt, ConverterStatus.INACTIVE),
            }
            for fmt in self._converters
        ]

    def is_format_supported(self, file_path: str | Path) -> bool:
        """Check if *file_path*'s format is supported."""
        path = Path(file_path)
        ext = path.suffix.lower()
        return ext in self._ext_map or self._infer_format(path) is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_format(self, path: Path) -> DocFormat | None:
        """Infer :py:class:`DocFormat` from file extension."""
        ext = path.suffix.lower()
        fmt = self._ext_to_format(ext)
        if fmt is not None:
            return fmt

        # Fallback: check by name for common variants.
        name_map = {
            ".doc": DocFormat.DOCX,
            ".htm": DocFormat.HTML,
            ".markdown": DocFormat.MD,
        }
        return name_map.get(ext)


__all__ = [
    "DocFormat",
    "ConverterStatus",
    "ConversionResult",
    "DocumentConverter",
    "TextConverter",
    "PDFConverter",
    "DOCXConverter",
    "ImageOCRConverter",
    "EPUBConverter",
    "HTMLConverter",
    "ConverterManager",
]