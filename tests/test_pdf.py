import pytest

from rag.errors import NoTextError, PdfError, PdfTooLargeError
from rag.pdf import fitz, read_pdf


def test_reads_every_page_of_the_sample(sample_bytes):
    pages = read_pdf(sample_bytes)
    assert [p.number for p in pages] == [1, 2, 3]
    assert "Why retrieval helps" in pages[0].text
    assert "Answering a question" in pages[1].text
    assert "Known limits" in pages[2].text


def test_headings_and_paragraphs_are_separate_blocks(sample_bytes):
    page = read_pdf(sample_bytes)[0]
    assert "\n\n" in page.text


def test_page_limit(sample_bytes):
    with pytest.raises(PdfTooLargeError, match="3 pages"):
        read_pdf(sample_bytes, max_pages=2)


def test_empty_file():
    with pytest.raises(PdfError, match="empty"):
        read_pdf(b"")


def test_not_a_pdf():
    with pytest.raises(PdfError, match="could not be read"):
        read_pdf(b"this is plain text, not a pdf")


def test_a_pdf_without_text_is_reported_as_a_scan():
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    with pytest.raises(NoTextError, match="OCR"):
        read_pdf(doc.tobytes())


def test_password_protected_pdf():
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret text")
    protected = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="open", owner_pw="own")
    with pytest.raises(PdfError, match="password"):
        read_pdf(protected)
