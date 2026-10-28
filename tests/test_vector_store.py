import re

import pytest
from fakes import DIM, ApiError, FakeIndex, FakePinecone, vector_of

from rag.chunking import Chunk
from rag.errors import ServiceError
from rag.vector_store import VectorStore


@pytest.fixture
def store(pinecone_client, settings, clock) -> VectorStore:
    return VectorStore("key", settings, client=pinecone_client, clock=clock, sleep=clock.sleep)


def chunk(i: int, text: str, page: int = 1) -> Chunk:
    return Chunk(f"c{i}", text, page, page)


def test_creates_the_index_once_with_the_expected_spec(store, pinecone_client, settings):
    store.ensure_index()
    store.ensure_index()
    store2 = VectorStore("key", settings, client=pinecone_client)
    store2.ensure_index()  # the index exists now, so nothing is created again
    assert pinecone_client.created == [
        {"name": settings.index_name, "dimension": 1024, "metric": "cosine", "cloud": "aws", "region": "us-east-1"}
    ]


def test_an_index_with_the_wrong_dimension_is_refused(settings, clock):
    pc = FakePinecone(dimension=768)
    pc.indexes[settings.index_name] = FakeIndex()
    with pytest.raises(ServiceError, match="dimension 768"):
        VectorStore("key", settings, client=pc, clock=clock, sleep=clock.sleep).ensure_index()


def test_a_conflict_while_creating_means_another_session_made_it(settings, clock):
    pc = FakePinecone()
    real_create = pc.create_index

    def create_then_conflict(**kwargs):
        real_create(**kwargs)
        raise ApiError(409, "already exists")

    pc.create_index = create_then_conflict
    VectorStore("key", settings, client=pc, clock=clock, sleep=clock.sleep).ensure_index()
    assert settings.index_name in pc.indexes


def test_other_creation_errors_are_raised(settings, clock):
    pc = FakePinecone()
    pc.fail_create_with = ApiError(403, "quota")
    with pytest.raises(ApiError):
        VectorStore("key", settings, client=pc, clock=clock, sleep=clock.sleep).ensure_index()


def test_upsert_then_query_returns_pages_and_text(store):
    chunks = [chunk(0, "cosine similarity compares vectors", 2), chunk(1, "cafeteria lunch menu", 5)]
    store.upsert("ns", chunks, [vector_of(c.text) for c in chunks], "demo.pdf")
    hits = store.query("ns", vector_of("how does cosine similarity work"), top_k=2)
    assert [h.chunk_id for h in hits] == ["c0", "c1"]
    assert hits[0].text == "cosine similarity compares vectors"
    assert (hits[0].page_start, hits[0].page_end) == (2, 2)
    assert hits[0].score > hits[1].score


def test_vectors_carry_the_filename_and_pages(store, pinecone_client):
    store.upsert("ns", [chunk(0, "some text here", 3)], [vector_of("some text here")], "report.pdf")
    _, metadata = pinecone_client.index("rag-document-qa").namespaces["ns"]["c0"]
    assert metadata == {"text": "some text here", "page_start": 3, "page_end": 3, "filename": "report.pdf"}


def test_namespaces_do_not_mix(store):
    store.upsert("a", [chunk(0, "alpha text")], [vector_of("alpha text")], "a.pdf")
    store.upsert("b", [chunk(0, "beta text")], [vector_of("beta text")], "b.pdf")
    assert [h.text for h in store.query("a", vector_of("alpha"), 5)] == ["alpha text"]


def test_wait_until_visible(store):
    store.upsert("ns", [chunk(0, "alpha text")], [vector_of("alpha text")], "a.pdf")
    assert store.wait_until_visible("ns", 1) is True


def test_wait_until_visible_gives_up(store, clock):
    start = clock.now
    assert store.wait_until_visible("missing", 5, timeout=3) is False
    assert clock.now - start >= 3


def test_deleting_a_missing_namespace_is_not_an_error(store):
    store.ensure_index()
    store.delete_namespace("never-existed")


def test_new_namespace_has_the_expected_shape(store):
    assert re.fullmatch(r"s\d{10}-[0-9a-f]{6}", store.new_namespace())


def test_purge_removes_only_old_namespaces_made_by_this_app(store, clock, pinecone_client):
    index = store.ensure_index()
    old = f"s{int(clock.now - 30 * 3600)}-aaaaaa"
    recent = f"s{int(clock.now - 3600)}-bbbbbb"
    for name in (old, recent, "someone-elses-namespace"):
        index.upsert(vectors=[{"id": "c0", "values": [0.0] * DIM, "metadata": {}}], namespace=name)

    assert store.purge_stale(max_age_hours=24) == 1
    assert set(index.namespaces) == {recent, "someone-elses-namespace"}
