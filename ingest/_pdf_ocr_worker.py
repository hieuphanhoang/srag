"""Subprocess worker: find scanned pages in a PDF and OCR them.

A page counts as scanned when it has (almost) no text layer but does contain
an image. When no page is scanned the worker stops there, and the caller keeps
using the markitdown CLI for the whole document (unchanged behavior for
ordinary PDFs). Otherwise the document is assembled page by page: text-layer
pages via pdfminer (the same engine markitdown uses), scanned pages rendered
with pypdfium2 and OCR'd with rapidocr.

All dependencies are permissively licensed (pdfminer.six MIT, pypdfium2
Apache-2.0/BSD-3, rapidocr Apache-2.0).

Run out-of-process (see ``_run_pdf_ocr_worker`` in converter.py) so a crash
or a runaway OCR pass on one bad PDF can be timed out and can't take the
whole ingest pipeline down with it — same rationale as the markitdown-CLI
subprocess routing for FD-24/25.

Usage: python _pdf_ocr_worker.py <pdf_path> <output_json_path>

Writes ``{"scanned_pages": [...], "text": str | null}`` to <output_json_path>
(``text`` is null when no page is scanned). The result goes to a file rather
than stdout because rapidocr prints its own status chatter to stdout. On
failure, prints the error to stderr and exits non-zero without writing the
output file.
"""

from __future__ import annotations

import json
import sys

# Pages with fewer non-whitespace characters than this in their text layer
# (e.g. only a page number or a stamp) are treated as having no text layer.
MIN_TEXT_CHARS = 10
OCR_DPI = 200


def _find_scanned_pages(pdf) -> list[int]:
    import pypdfium2.raw as pdfium_c

    scanned = []
    for index in range(len(pdf)):
        page = pdf[index]
        text = page.get_textpage().get_text_bounded()
        if len("".join(text.split())) >= MIN_TEXT_CHARS:
            continue
        if any(True for _ in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])):
            scanned.append(index)
    return scanned


def _text_layer_pages(pdf_path: str, page_count: int) -> list[str]:
    from pdfminer.high_level import extract_text

    # pdfminer ends every page with a form feed.
    pages = extract_text(pdf_path).split("\f")
    if len(pages) >= page_count:
        return pages[:page_count]
    return [extract_text(pdf_path, page_numbers=[i]) for i in range(page_count)]


def _ocr_page(engine, page) -> str:
    import numpy as np

    image = page.render(scale=OCR_DPI / 72).to_pil().convert("RGB")
    result = engine(np.asarray(image))
    return "\n".join(result.txts) if result is not None and result.txts else ""


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _pdf_ocr_worker.py <pdf_path> <output_json_path>", file=sys.stderr)
        return 2

    pdf_path, out_path = sys.argv[1], sys.argv[2]

    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        scanned = _find_scanned_pages(pdf)
        text = None
        if scanned:
            from rapidocr import RapidOCR

            engine = RapidOCR()
            scanned_set = set(scanned)
            text_pages = _text_layer_pages(pdf_path, len(pdf))
            parts = []
            for index in range(len(pdf)):
                page_text = _ocr_page(engine, pdf[index]) if index in scanned_set else text_pages[index]
                if page_text.strip():
                    parts.append(page_text.strip())
            text = "\n\n".join(parts)
    except Exception as exc:
        print(f"PDF OCR failed: {exc}", file=sys.stderr)
        return 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scanned_pages": scanned, "text": text}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
