import pytest
from fakes import ApiError, FakeIndex

from rag.cohere_service import CohereService
from rag.config import Settings
from rag.errors import NoTextError, PdfTooLargeError
from rag.pdf import fitz
from rag.pipeline import RAGPipeline, file_digest
from rag.prompts import NOT_FOUND
from rag.vector_store import VectorStore


def make_pipeline(settings, cohere_client, pinecone_client, clock) -> RAGPipeline:
    return RAGPipeline(
        CohereService("key", settings, client=cohere_client, sleep=clock.sleep),
        VectorStore("key", settings, client=pinecone_client, clock=clock, sleep=clock.sleep),
        settings,
    )


def test_ingest_reads_chunks_embeds_and_stores(pipeline, sample_bytes, cohere_client, pinecone_client, settings):
    doc = pipeline.ingest(sample_bytes, "how-rag-works.pdf")
    assert doc.filename == "how-rag-works.pdf" and doc.pages == 3 and doc.chunk_count >= 3
    assert doc.digest == file_digest(sample_bytes)

    stored = pinecone_client.index(settings.index_name).namespaces[doc.namespace]
    assert len(stored) == doc.chunk_count
    pages = {meta["page_start"] for _, meta in stored.values()}
    assert pages == {1, 2, 3}
    assert {c["input_type"] for c in cohere_client.calls_to("embed")} == {"search_document"}


def test_ingest_reports_each_step(pipeline, sample_bytes):
    steps = []
    pipeline.ingest(sample_bytes, "demo.pdf", progress=steps.append)
    assert steps[0] == "Read 3 pages"
    assert any(s.startswith("Split the text into") for s in steps)
    assert steps[-1] == "Storing the vectors in Pinecone"


def test_two_uploads_get_separate_namespaces(pipeline, sample_bytes, pinecone_client, settings):
    first = pipeline.ingest(sample_bytes, "a.pdf")
    second = pipeline.ingest(sample_bytes, "b.pdf")
    assert first.namespace != second.namespace
    assert len(pinecone_client.index(settings.index_name).namespaces) == 2


def test_the_question_finds_the_chunk_that_has_the_answer(pipeline, sample_bytes):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    answer = pipeline.ask(doc.namespace, "How many numbers are in an embedding?")
    assert any("1,024 numbers" in s.text for s in answer.sources)
    assert answer.text.startswith("According to the document")
    assert answer.searched == "How many numbers are in an embedding?"


def test_sources_carry_pages_scores_and_the_cited_flag(pipeline, sample_bytes):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    answer = pipeline.ask(doc.namespace, "What kind of PDF needs OCR first?")
    assert len(answer.sources) == 3
    top = answer.sources[0]
    assert top.cited and not answer.sources[1].cited
    assert top.page_start >= 1 and 0 <= top.score <= 1
    assert "scanned" in top.text.lower()


def test_search_and_rerank_use_the_configured_sizes(pipeline, sample_bytes, cohere_client, pinecone_client, settings):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    pipeline.ask(doc.namespace, "What is cosine similarity?")
    (rerank,) = cohere_client.calls_to("rerank")
    assert rerank["top_n"] == settings.rerank_top_n == 3
    assert len(rerank["documents"]) <= settings.retrieve_top_k == 10


def test_a_follow_up_is_rewritten_before_searching(pipeline, sample_bytes, cohere_client):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    history = [("user", "How are chunks made?"), ("assistant", "From sentences.")]
    answer = pipeline.ask(doc.namespace, "and the overlap?", history)
    assert answer.searched == "REWRITTEN and the overlap?"
    assert cohere_client.calls_to("embed")[-1]["texts"] == ["REWRITTEN and the overlap?"]


def test_a_first_question_is_not_rewritten(pipeline, sample_bytes, cohere_client):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    pipeline.ask(doc.namespace, "What is a chunk?")
    assert [c for c in cohere_client.calls_to("chat") if c["documents"] is None] == []


def test_rewriting_can_be_switched_off(sample_bytes, cohere_client, pinecone_client, clock):
    pipe = make_pipeline(Settings(rewrite_followups=False), cohere_client, pinecone_client, clock)
    doc = pipe.ingest(sample_bytes, "demo.pdf")
    answer = pipe.ask(doc.namespace, "and the overlap?", [("user", "a"), ("assistant", "b")])
    assert answer.searched == "and the overlap?"


def test_weak_matches_are_dropped_and_the_model_is_not_called(sample_bytes, cohere_client, pinecone_client, clock):
    pipe = make_pipeline(Settings(min_relevance=0.99), cohere_client, pinecone_client, clock)
    doc = pipe.ingest(sample_bytes, "demo.pdf")
    answer = pipe.ask(doc.namespace, "How many numbers are in an embedding?")
    assert answer.text == NOT_FOUND and answer.sources == []
    assert [c for c in cohere_client.calls_to("chat") if c["documents"]] == []


def test_an_unknown_namespace_gives_the_not_found_answer(pipeline, sample_bytes, cohere_client):
    pipeline.ingest(sample_bytes, "demo.pdf")
    answer = pipeline.ask("no-such-namespace", "anything?")
    assert answer.text == NOT_FOUND
    assert cohere_client.calls_to("rerank") == []


def test_clear_removes_the_vectors(pipeline, sample_bytes, pinecone_client, settings):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    pipeline.clear(doc.namespace)
    assert doc.namespace not in pinecone_client.index(settings.index_name).namespaces


def test_clearing_twice_does_not_raise(pipeline, sample_bytes):
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    pipeline.clear(doc.namespace)
    pipeline.clear(doc.namespace)


def test_a_failed_upload_leaves_nothing_behind(pipeline, sample_bytes, pinecone_client, settings, monkeypatch):
    def boom(self, **kwargs):
        raise ApiError(500, "pinecone is down")

    monkeypatch.setattr(FakeIndex, "upsert", boom)
    with pytest.raises(ApiError):
        pipeline.ingest(sample_bytes, "demo.pdf")
    assert pinecone_client.index(settings.index_name).namespaces == {}


def test_a_document_over_the_chunk_limit_is_refused_before_any_api_call(sample_bytes, cohere_client, pinecone_client, clock):
    pipe = make_pipeline(Settings(max_chunks=2), cohere_client, pinecone_client, clock)
    with pytest.raises(PdfTooLargeError, match="limit is 2"):
        pipe.ingest(sample_bytes, "demo.pdf")
    assert cohere_client.calls == [] and pinecone_client.created == []


def test_a_scan_is_refused_before_any_api_call(pipeline, cohere_client, pinecone_client):
    doc = fitz.open()
    doc.new_page()
    with pytest.raises(NoTextError):
        pipeline.ingest(doc.tobytes(), "scan.pdf")
    assert cohere_client.calls == [] and pinecone_client.created == []


def test_old_namespaces_are_purged_after_an_upload(pipeline, sample_bytes, pinecone_client, settings, clock):
    index = pipeline.store.ensure_index()
    old = f"s{int(clock.now - 48 * 3600)}-abcdef"
    index.upsert(vectors=[{"id": "c0", "values": [0.0] * 1024, "metadata": {}}], namespace=old)
    doc = pipeline.ingest(sample_bytes, "demo.pdf")
    assert set(index.namespaces) == {doc.namespace}


def test_a_failure_while_purging_does_not_fail_the_upload(pipeline, sample_bytes, monkeypatch):
    monkeypatch.setattr(pipeline.store, "purge_stale", lambda hours: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pipeline.ingest(sample_bytes, "demo.pdf").chunk_count > 0
