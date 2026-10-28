import pytest
from fakes import sdk_error

from rag.config import Settings, settings_from_env
from rag.errors import NoTextError, RagError, explain, scrub
from rag.keys import load_keys, looks_like_placeholder
from rag.render import neutralize_links


@pytest.mark.parametrize(
    "module, status, expected",
    [
        ("cohere.errors", 401, "Cohere rejected the API key"),
        ("pinecone.exceptions", 401, "Pinecone rejected the API key"),
        ("pinecone.exceptions", 403, "no room for another index"),
        ("cohere.errors", 403, "Cohere refused the request"),
        ("cohere.errors", 404, "could not find the model or index"),
        ("cohere.errors", 429, "rate limiting this key"),
        ("pinecone.exceptions", 503, "Pinecone is having problems (HTTP 503)"),
    ],
)
def test_service_errors_are_explained(module, status, expected):
    assert expected in explain(sdk_error(module, status))


def test_our_own_errors_are_shown_as_they_are():
    assert explain(NoTextError("This PDF has no selectable text.")) == "This PDF has no selectable text."
    assert isinstance(NoTextError("x"), RagError)


def test_a_timeout_is_recognised():
    assert "timed out" in explain(TimeoutError("slow"))


def test_unknown_errors_name_the_type_and_hide_anything_key_shaped():
    message = explain(ValueError("bad value for token token-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 in request"))
    assert "ValueError" in message and "<hidden>" in message
    assert "token-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456" not in message


def test_scrub_shortens_long_text():
    assert len(scrub("word " * 200)) == 160


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain answer with no links", "plain answer with no links"),
        ("see [the paper](https://example.com/a?x=1) now", "see the paper (https:​//example.com/a?x=1) now"),
        ("![chart](https://evil.example/p.png?d=secret)", "!chart (https:​//evil.example/p.png?d=secret)"),
        ("visit https://example.com or http://example.org", "visit https:​//example.com or http:​//example.org"),
    ],
)
def test_links_and_images_are_shown_as_plain_text(text, expected):
    assert neutralize_links(text) == expected


def test_defaults_match_the_original_project():
    s = Settings()
    assert (s.chunk_size, s.retrieve_top_k, s.rerank_top_n) == (1000, 10, 3)
    assert (s.embed_model, s.rerank_model, s.chat_model) == ("embed-english-v3.0", "rerank-english-v3.0", "command-r-plus-08-2024")


@pytest.mark.parametrize(
    "kwargs",
    [{"chunk_size": 100}, {"chunk_overlap": 600}, {"rerank_top_n": 0}, {"retrieve_top_k": 2, "rerank_top_n": 3}, {"min_relevance": 1.5}],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_can_be_overridden_from_the_environment():
    s = settings_from_env({
        "RAG_CHUNK_SIZE": "800", "RAG_CHUNK_OVERLAP": "100", "RAG_MIN_RELEVANCE": "0.25", "RAG_REWRITE_FOLLOWUPS": "false",
        "COHERE_CHAT_MODEL": "command-a-03-2025", "PINECONE_INDEX_NAME": "my-index", "RAG_MAX_QUESTIONS": 12,
    })
    assert (s.chunk_size, s.chunk_overlap, s.min_relevance) == (800, 100, 0.25)
    assert s.rewrite_followups is False and s.chat_model == "command-a-03-2025"
    assert s.index_name == "my-index" and s.max_questions == 12


def test_empty_environment_values_keep_the_defaults():
    assert settings_from_env({"RAG_CHUNK_SIZE": "", "COHERE_CHAT_MODEL": ""}) == Settings()


def test_keys_come_from_the_environment_first(tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('COHERE_API_KEY = "from-file-cohere"\nPINECONE_API_KEY = "from-file-pinecone"\n', encoding="utf-8")
    env = {"COHERE_API_KEY": "from-env-cohere"}
    assert load_keys(env, secrets) == ("from-env-cohere", "from-file-pinecone")
    assert load_keys({}, None) == (None, None)


def test_a_missing_or_broken_secrets_file_is_ignored(tmp_path):
    assert load_keys({}, tmp_path / "nope.toml") == (None, None)
    broken = tmp_path / "broken.toml"
    broken.write_text("this is = not [valid toml", encoding="utf-8")
    assert load_keys({}, broken) == (None, None)


@pytest.mark.parametrize("value, expected", [(None, True), ("", True), ("your_cohere_api_key_here", True), ("short", True),
                                              ("aB3dE5gH7jK9mN1pQ3sT5vW7yZ9", False)])
def test_placeholder_detection(value, expected):
    assert looks_like_placeholder(value) is expected
