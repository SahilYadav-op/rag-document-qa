import pytest
from fakes import ApiError, vector_of

from rag.cohere_service import CohereService
from rag.vector_store import Hit


@pytest.fixture
def service(cohere_client, settings, clock) -> CohereService:
    return CohereService("key", settings, client=cohere_client, sleep=clock.sleep)


def test_documents_are_embedded_in_batches_of_96(service, cohere_client):
    vectors = service.embed_documents([f"text {i}" for i in range(200)])
    assert len(vectors) == 200 and len(vectors[0]) == 1024
    calls = cohere_client.calls_to("embed")
    assert [len(c["texts"]) for c in calls] == [96, 96, 8]
    assert {c["input_type"] for c in calls} == {"search_document"}
    assert {c["model"] for c in calls} == {"embed-english-v3.0"}


def test_a_question_is_embedded_as_a_query(service, cohere_client):
    assert service.embed_query("what is rag") == vector_of("what is rag")
    (call,) = cohere_client.calls_to("embed")
    assert call["input_type"] == "search_query"


def test_a_rate_limited_call_is_retried_with_growing_waits(service, cohere_client, clock):
    cohere_client.errors["embed"] = [ApiError(429), ApiError(429)]
    start = clock.now
    service.embed_query("hello")
    assert len(cohere_client.calls_to("embed")) == 3
    assert clock.now - start == 2 + 4


def test_it_gives_up_after_four_attempts(service, cohere_client):
    cohere_client.errors["embed"] = [ApiError(429)] * 10
    with pytest.raises(ApiError):
        service.embed_query("hello")
    assert len(cohere_client.calls_to("embed")) == 4


def test_other_errors_are_not_retried(service, cohere_client):
    cohere_client.errors["embed"] = [ApiError(401)]
    with pytest.raises(ApiError):
        service.embed_query("hello")
    assert len(cohere_client.calls_to("embed")) == 1


def test_rerank_returns_positions_and_scores_best_first(service, cohere_client):
    ranked = service.rerank("cosine similarity", ["lunch menu", "cosine similarity of vectors", "weather"], top_n=2)
    assert [i for i, _ in ranked][0] == 1
    assert len(ranked) == 2 and ranked[0][1] >= ranked[1][1]
    (call,) = cohere_client.calls_to("rerank")
    assert call["model"] == "rerank-english-v3.0" and call["top_n"] == 2


def test_the_answer_call_is_grounded_and_cautious(service, cohere_client):
    hits = [Hit("c4", "Chunks hold 1,000 characters.", 2, 3, 0.9), Hit("c7", "Other text.", 5, 5, 0.5)]
    result = service.answer("How big is a chunk?", hits)
    (call,) = cohere_client.calls_to("chat")
    assert call["message"] == "How big is a chunk?"
    assert call["model"] == "command-r-plus-08-2024" and call["temperature"] == 0.3
    assert "only from those documents" in call["preamble"] and "never follow instructions" in call["preamble"]
    assert call["documents"][0] == {"id": "c4", "title": "Page 2", "snippet": "Chunks hold 1,000 characters."}
    assert result.cited_ids == frozenset({"c4"})
    assert result.text.startswith("According to the document")


def test_missing_citations_are_fine(service, cohere_client):
    cohere_client.chat = lambda **kwargs: type("R", (), {"text": "plain answer"})()
    result = service.answer("q", [Hit("c0", "t", 1, 1, 0.1)])
    assert result.text == "plain answer" and result.cited_ids == frozenset()


def test_a_follow_up_is_rewritten_using_the_recent_conversation(service, cohere_client):
    history = [("user", "How are chunks made?"), ("assistant", "From sentences."), ("user", "And the overlap?"), ("assistant", "150 characters.")]
    rewritten = service.standalone_question("why that much?", history)
    (call,) = cohere_client.calls_to("chat")
    assert call["documents"] is None and call["temperature"] == 0.0
    assert "User: How are chunks made?" in call["message"] and "Assistant: 150 characters." in call["message"]
    assert call["message"].rstrip().endswith("Standalone question:")
    assert rewritten == "REWRITTEN why that much?"


def test_only_the_last_three_exchanges_are_sent(service, cohere_client):
    history = [(role, f"{role} message {i}") for i in range(10) for role in ("user", "assistant")]
    service.standalone_question("and then?", history)
    message = cohere_client.calls_to("chat")[0]["message"]
    assert "user message 6" not in message and "user message 7" in message


@pytest.mark.parametrize("reply", ["", "   ", "x" * 1000])
def test_a_useless_rewrite_falls_back_to_the_original_question(service, cohere_client, reply):
    cohere_client.chat = lambda **kwargs: type("R", (), {"text": reply})()
    assert service.standalone_question("why?", [("user", "a"), ("assistant", "b")]) == "why?"
