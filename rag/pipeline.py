"""Ingest a PDF and answer questions about it: read, chunk, embed, store, then embed the question, search, rerank, answer."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .chunking import chunk_pages
from .cohere_service import CohereService
from .config import Settings
from .errors import NoTextError, PdfTooLargeError
from .pdf import read_pdf
from .prompts import NOT_FOUND
from .vector_store import VectorStore

log = logging.getLogger(__name__)


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass(frozen=True)
class Document:
    filename: str
    namespace: str
    pages: int
    chunk_count: int
    digest: str


@dataclass(frozen=True)
class Source:
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    score: float
    cited: bool  # the model pointed at this chunk in its answer


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    searched: str  # the question that was actually searched for (differs from the user's for follow-ups)


class RAGPipeline:
    def __init__(self, cohere_service: CohereService, store: VectorStore, settings: Settings):
        self.cohere = cohere_service
        self.store = store
        self.settings = settings

    def ingest(self, data: bytes, filename: str, progress: Callable[[str], None] | None = None) -> Document:
        say = progress or (lambda message: None)
        s = self.settings

        pages = read_pdf(data, s.max_pages)
        say(f"Read {len(pages)} pages")
        chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)
        if not chunks:
            raise NoTextError("No usable text was found in this PDF.")
        if len(chunks) > s.max_chunks:
            raise PdfTooLargeError(f"This PDF makes {len(chunks)} chunks. The limit is {s.max_chunks}.")
        say(f"Split the text into {len(chunks)} chunks")

        say("Preparing the Pinecone index (the first time can take a minute)")
        self.store.ensure_index()
        say("Embedding the chunks with Cohere")
        vectors = self.cohere.embed_documents([c.text for c in chunks])

        namespace = self.store.new_namespace()
        say("Storing the vectors in Pinecone")
        try:
            self.store.upsert(namespace, chunks, vectors, filename)
            self.store.wait_until_visible(namespace, len(chunks))
        except Exception:
            self.clear(namespace)  # do not leave a half-filled namespace behind
            raise

        try:
            self.store.purge_stale(s.namespace_ttl_hours)
        except Exception:  # housekeeping must never fail an upload
            log.warning("Could not remove old namespaces", exc_info=True)
        return Document(filename, namespace, len(pages), len(chunks), file_digest(data))

    def ask(self, namespace: str, question: str, history: Sequence[tuple[str, str]] = ()) -> Answer:
        s = self.settings
        searched = question
        if history and s.rewrite_followups:
            searched = self.cohere.standalone_question(question, history)

        hits = self.store.query(namespace, self.cohere.embed_query(searched), s.retrieve_top_k)
        if not hits:
            return Answer(NOT_FOUND, [], searched)

        ranked = self.cohere.rerank(searched, [h.text for h in hits], s.rerank_top_n)
        kept = [(hits[i], score) for i, score in ranked if score >= s.min_relevance]
        if not kept:
            return Answer(NOT_FOUND, [], searched)

        generated = self.cohere.answer(searched, [hit for hit, _ in kept])
        sources = [
            Source(h.chunk_id, h.page_start, h.page_end, h.text, score, h.chunk_id in generated.cited_ids)
            for h, score in kept
        ]
        return Answer(generated.text or NOT_FOUND, sources, searched)

    def clear(self, namespace: str) -> None:
        try:
            self.store.delete_namespace(namespace)
        except Exception:
            log.warning("Could not delete namespace %s", namespace, exc_info=True)


def build_pipeline(cohere_key: str, pinecone_key: str, settings: Settings) -> RAGPipeline:
    return RAGPipeline(CohereService(cohere_key, settings), VectorStore(pinecone_key, settings), settings)
