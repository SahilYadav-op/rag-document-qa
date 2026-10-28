"""Read a PDF into pages of text with PyMuPDF."""
from __future__ import annotations

from dataclasses import dataclass

from .errors import NoTextError, PdfError, PdfTooLargeError

try:  # PyMuPDF 1.24.3 and later export the pymupdf name; older releases only have fitz
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as printed in the viewer
    text: str


def read_pdf(data: bytes, max_pages: int = 80) -> list[Page]:
    """Return the text of every page. Raises PdfError (or a subclass) with a message fit to show the user."""
    if not data:
        raise PdfError("The file is empty.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types for bad input
        raise PdfError("This file could not be read as a PDF.") from exc
    with doc:
        if doc.needs_pass:
            raise PdfError("This PDF is password protected.")
        if doc.page_count > max_pages:
            raise PdfTooLargeError(f"This PDF has {doc.page_count} pages. The limit is {max_pages}.")
        pages = [Page(i + 1, _page_text(page)) for i, page in enumerate(doc)]
    if not any(p.text.strip() for p in pages):
        raise NoTextError("This PDF has no selectable text, so it is probably a scan. It needs OCR first.")
    return pages


def _page_text(page) -> str:
    # Blocks keep paragraphs apart, which plain text mode does not. They stay in the order the PDF stores them,
    # which follows the reading order of well-formed files (sorting by position would interleave columns).
    blocks = page.get_text("blocks")
    return "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
