"""Generate samples/how-rag-works.pdf: the small document behind the "Use the sample PDF" button, the tests and the eval example.

Run from the repository root:  python scripts/make_sample_pdf.py
"""
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

OUT = Path(__file__).resolve().parent.parent / "samples" / "how-rag-works.pdf"

PAGES = [
    [
        ("title", "How retrieval-augmented generation works"),
        ("note", "A short explainer that ships with the RAG Document QA app."),
        ("heading", "1. Why retrieval helps"),
        ("body", "A language model only knows what was in its training data. It has never seen your documents, and it "
                 "can answer with confidence even when it is wrong. Retrieval-augmented generation, or RAG, fixes this "
                 "by finding the relevant passages first and then asking the model to answer from them. The model is "
                 "the student in an open-book exam, and the passages are the pages of the book it is allowed to use."),
        ("heading", "2. Indexing a document"),
        ("body", "Indexing happens once for each document. The app reads the text of every page and splits it into "
                 "chunks. A chunk holds about 1,000 characters, and neighbouring chunks overlap by 150 characters so "
                 "that a sentence on a boundary is never lost. Each chunk is turned into an embedding, a list of 1,024 "
                 "numbers that captures its meaning. The embeddings are stored in a vector database together with the "
                 "chunk text and its page number."),
    ],
    [
        ("heading", "3. Answering a question"),
        ("body", "For every question the app embeds the question with the same model and asks the vector database for "
                 "the ten nearest chunks. Nearness is measured with cosine similarity, which compares the direction of "
                 "two vectors. Vector search is fast but rough, so a reranker then reads the question together with "
                 "each of the ten chunks and keeps the best three. Those three chunks are sent to the language model "
                 "with an instruction to answer only from them. The answer is shown together with the pages it used."),
        ("heading", "4. Follow-up questions"),
        ("body", "A follow-up such as \"what about the second one?\" means nothing on its own. When the conversation "
                 "already has earlier turns, the app first asks the model to rewrite the follow-up as a standalone "
                 "question, and it searches with the rewritten question."),
    ],
    [
        ("heading", "5. Known limits"),
        ("body", "A scanned PDF has no selectable text, so it needs OCR before it can be used. Tables are read as a "
                 "stream of words and lose their layout. A document is untrusted input: a line inside it may try to "
                 "give the model instructions, so the app tells the model to ignore instructions found in documents "
                 "and shows links and images in answers as plain text."),
        ("note", "End of the sample document."),
    ],
]

STYLES = {  # kind -> (font, size, space after)
    "title": ("hebo", 20, 8),
    "note": ("helv", 10, 18),
    "heading": ("hebo", 14, 6),
    "body": ("helv", 11, 18),
}


def write(page, y, kind, text):
    font, size, gap = STYLES[kind]
    box = fitz.Rect(56, y, 539, 780)
    left = page.insert_textbox(box, text, fontsize=size, fontname=font, lineheight=1.35)
    if left < 0:
        raise RuntimeError(f"text does not fit on the page: {text[:40]!r}")
    return y + (box.height - left) + gap


def main():
    doc = fitz.open()
    for items in PAGES:
        page = doc.new_page(width=595, height=842)
        y = 60
        for kind, text in items:
            y = write(page, y, kind, text)
    doc.set_metadata({"title": "How retrieval-augmented generation works", "author": "Sahil Yadav"})
    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT, deflate=True, garbage=3)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(doc)} pages)")


if __name__ == "__main__":
    main()
