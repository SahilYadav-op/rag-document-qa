import re

import pytest

from rag.chunking import chunk_pages, split_sentences
from rag.pdf import Page


def numbered_sentences(count: int) -> str:
    return " ".join(f"Sentence {i:03d} talks about retrieval and chunking in plain words for the reader." for i in range(count))


def ids_in(text: str) -> list[str]:
    return re.findall(r"Sentence (\d{3})", text)


def test_split_sentences_basic():
    assert split_sentences("First one. Second one! Third? Fourth.") == ["First one.", "Second one!", "Third?", "Fourth."]


def test_abbreviations_do_not_end_a_sentence():
    text = "Use a vector store, e.g. Pinecone, for this. See Fig. 3 for details. Dr. Smith agrees."
    assert split_sentences(text) == [
        "Use a vector store, e.g. Pinecone, for this.",
        "See Fig. 3 for details.",
        "Dr. Smith agrees.",
    ]


def test_decimals_and_initials_stay_inside_a_sentence():
    assert split_sentences("The score was 3.5 overall. J. Smith wrote it.") == ["The score was 3.5 overall.", "J. Smith wrote it."]


def test_chunks_respect_the_size_limit():
    chunks = chunk_pages([Page(1, numbered_sentences(60))], chunk_size=400, overlap=120)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)


def test_the_last_sentence_of_a_chunk_starts_the_next_one():
    chunks = chunk_pages([Page(1, numbered_sentences(60))], chunk_size=400, overlap=120)
    for before, after in zip(chunks, chunks[1:]):
        assert ids_in(before.text)[-1] == ids_in(after.text)[0]


def test_no_overlap_when_it_is_zero():
    chunks = chunk_pages([Page(1, numbered_sentences(60))], chunk_size=400, overlap=0)
    for before, after in zip(chunks, chunks[1:]):
        assert int(ids_in(after.text)[0]) == int(ids_in(before.text)[-1]) + 1


def test_every_sentence_lands_in_some_chunk():
    chunks = chunk_pages([Page(1, numbered_sentences(60))], chunk_size=400, overlap=120)
    seen = {i for c in chunks for i in ids_in(c.text)}
    assert seen == {f"{i:03d}" for i in range(60)}


def test_chunk_ids_are_sequential():
    chunks = chunk_pages([Page(1, numbered_sentences(40))], chunk_size=300, overlap=0)
    assert [c.id for c in chunks] == [f"c{i}" for i in range(len(chunks))]


def test_chunks_record_the_pages_they_touch():
    pages = [Page(1, "Alpha sentence about retrieval on the first page of the file."),
             Page(2, "Beta sentence about reranking on the second page of the file.")]
    (only,) = chunk_pages(pages, chunk_size=1000, overlap=0)
    assert (only.page_start, only.page_end) == (1, 2)

    first, second = chunk_pages(pages, chunk_size=100, overlap=0)
    assert (first.page_start, first.page_end) == (1, 1)
    assert (second.page_start, second.page_end) == (2, 2)


def test_a_sentence_repeated_for_overlap_does_not_widen_the_page_range():
    page_one = numbered_sentences(5)
    page_two = " ".join(f"Sentence {i:03d} talks about retrieval and chunking in plain words for the reader." for i in range(5, 10))
    first, second = chunk_pages([Page(1, page_one), Page(2, page_two)], chunk_size=400, overlap=100)[:2]
    assert second.text.startswith("Sentence 004")  # carried over from page 1
    assert (first.page_start, first.page_end) == (1, 1)
    assert (second.page_start, second.page_end) == (2, 2)


def test_a_sentence_longer_than_a_chunk_is_cut_at_word_boundaries():
    chunks = chunk_pages([Page(1, "word " * 500)], chunk_size=400, overlap=0)
    assert len(chunks) >= 6
    assert all(len(c.text) <= 400 for c in chunks)
    assert sum(c.text.count("word") for c in chunks) == 500


def test_a_token_longer_than_a_chunk_is_cut_hard():
    chunks = chunk_pages([Page(1, "x" * 1000)], chunk_size=300, overlap=0)
    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 1000


def test_page_numbers_and_other_crumbs_are_dropped():
    assert chunk_pages([Page(1, "12"), Page(2, "Page 3")]) == []


def test_words_hyphenated_at_a_line_break_are_joined():
    (chunk,) = chunk_pages([Page(1, "Retrieval-augmen-\ntation works well for long documents.")])
    assert "augmentation" in chunk.text


def test_wrapped_lines_become_one_sentence_but_paragraphs_stay_apart():
    text = "This sentence is wrapped\nacross two lines and continues.\n\nA second paragraph starts here."
    (chunk,) = chunk_pages([Page(1, text)])
    assert "wrapped across two lines" in chunk.text
    assert "continues. A second paragraph" in chunk.text


@pytest.mark.parametrize("size, overlap", [(50, 0), (1000, 600), (1000, -1)])
def test_bad_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], chunk_size=size, overlap=overlap)
