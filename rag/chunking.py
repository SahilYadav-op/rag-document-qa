"""Split page text into overlapping, sentence-aware chunks that remember which pages they came from."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .pdf import Page


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int


# a sentence ends at . ! or ? (plus closing quotes or brackets) followed by space and something that starts a sentence
_BOUNDARY = re.compile(r"""([.!?]["')\]]*)\s+(?=["'(\[]?[A-Z0-9])""")
# "e.g.", "Fig.", "Dr." and a lone initial such as "J." do not end a sentence
_ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|fig|figs|eq|dr|mr|mrs|ms|prof|no|al|approx|cf|inc|ltd|st)|\b[A-Z])\.$",
    re.IGNORECASE,
)

Unit = tuple[str, int]  # one sentence and the page it is on


def split_sentences(paragraph: str) -> list[str]:
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(paragraph):
        candidate = paragraph[start:match.end(1)]
        if _ABBREVIATION.search(candidate):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _paragraphs(text: str) -> list[str]:
    text = re.sub(r"-\n(?=[a-z])", "", text.replace("\r", ""))  # join words that were hyphenated at a line break
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s*\n\s*", " ", block)  # a single line break inside a paragraph is only wrapping
        block = re.sub(r"[ \t ]{2,}", " ", block).strip()
        if block:
            paragraphs.append(block)
    return paragraphs


def _fit(sentence: str, size: int) -> list[str]:
    """Cut a sentence that is longer than a chunk at word boundaries."""
    if len(sentence) <= size:
        return [sentence]
    pieces, current = [], ""
    for word in sentence.split():
        while len(word) > size:  # a single token longer than a chunk, such as a URL or a table row
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _length(units: Sequence[Unit]) -> int:
    return sum(len(text) for text, _ in units) + max(len(units) - 1, 0)


def _tail(units: Sequence[Unit], overlap: int) -> list[Unit]:
    """The last sentences of a chunk that fit inside the overlap budget."""
    tail: list[Unit] = []
    total = 0
    for unit in reversed(units):
        add = len(unit[0]) + (1 if tail else 0)
        if total + add > overlap:
            break
        tail.insert(0, unit)
        total += add
    return tail


def chunk_pages(pages: Sequence[Page], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Pack sentences into chunks of at most chunk_size characters.

    Each chunk starts with the last sentences of the previous one (up to `overlap` characters), so a fact that sits on a
    boundary appears whole in at least one chunk. A chunk records the first and last page it touches.
    """
    if chunk_size < 100 or not 0 <= overlap <= chunk_size // 2:
        raise ValueError("chunk_size must be at least 100 and overlap at most half of it")

    units: list[Unit] = [
        (piece, page.number)
        for page in pages
        for paragraph in _paragraphs(page.text)
        for sentence in split_sentences(paragraph)
        for piece in _fit(sentence, chunk_size)
    ]

    groups: list[tuple[list[Unit], int]] = []  # (sentences, how many of them were carried over from the previous chunk)
    current: list[Unit] = []
    carried = 0
    new = 0  # sentences added since the last chunk was closed, so a tail-only chunk is never emitted
    for unit in units:
        if current and _length(current) + 1 + len(unit[0]) > chunk_size:
            groups.append((current, carried))
            current = _tail(current, overlap)
            while current and _length(current) + 1 + len(unit[0]) > chunk_size:
                current = current[1:]
            carried, new = len(current), 0
        current = [*current, unit]
        new += 1
    if new:
        groups.append((current, carried))

    chunks: list[Chunk] = []
    for group, carried in groups:
        text = " ".join(t for t, _ in group)
        if len(re.sub(r"\W", "", text)) < 20:  # page numbers, running headers and other crumbs
            continue
        # the pages come from the new sentences only, so a sentence repeated for overlap does not widen the range
        numbers = [n for _, n in group[carried:]]
        chunks.append(Chunk(f"c{len(chunks)}", text, min(numbers), max(numbers)))
    return chunks
