from pathlib import Path

import pytest
import streamlit as st
from fakes import FakeUpload, sdk_error
from streamlit.testing.v1 import AppTest

from rag.pdf import fitz

APP = str(Path(__file__).resolve().parent.parent / "app.py")
BUILTIN = {"COHERE_API_KEY": "c" * 30, "PINECONE_API_KEY": "p" * 30}


@pytest.fixture(autouse=True)
def fake_services(monkeypatch, cohere_client, pinecone_client):
    monkeypatch.setattr("cohere.Client", lambda api_key: cohere_client)
    monkeypatch.setattr("rag.vector_store.Pinecone", lambda api_key: pinecone_client)
    for name in ("COHERE_API_KEY", "PINECONE_API_KEY", "RAG_MAX_QUESTIONS"):
        monkeypatch.delenv(name, raising=False)


def start(secrets: dict | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    for name, value in (secrets or {}).items():
        at.secrets[name] = value
    return at.run()


def add_keys(at: AppTest) -> AppTest:
    at.text_input(key="cohere_key").set_value("c" * 30)
    at.text_input(key="pinecone_key").set_value("p" * 30)
    return at.run()


def load_sample(at: AppTest) -> AppTest:
    return at.button(key="use_sample").click().run()


def ask(at: AppTest, question: str) -> AppTest:
    return at.chat_input[0].set_value(question).run()


def test_the_page_starts_and_asks_for_keys():
    at = start()
    assert not at.exception
    assert at.title[0].value == "RAG Document QA"
    assert any("Add your Cohere and Pinecone API keys" in w.value for w in at.warning)
    assert at.chat_input[0].disabled


def test_the_sample_pdf_is_ingested_and_a_question_is_answered(pinecone_client):
    at = load_sample(add_keys(start()))
    assert not at.exception
    assert any("how-rag-works.pdf: 3 pages" in c.value for c in at.caption)
    assert not at.chat_input[0].disabled
    assert len(pinecone_client.indexes["rag-document-qa"].namespaces) == 1

    at = ask(at, "How many numbers are in an embedding?")
    assert not at.exception
    assert len(at.chat_message) == 2
    assert "According to the document" in at.chat_message[1].markdown[0].value
    assert at.session_state["questions"] == 1
    assert at.session_state["messages"][-1]["sources"]


def test_a_follow_up_question_is_searched_in_rewritten_form():
    at = load_sample(add_keys(start()))
    at = ask(at, "How many numbers are in an embedding?")
    at = ask(at, "and the overlap?")
    assert not at.exception
    assert at.session_state["messages"][-1]["searched"] == "REWRITTEN and the overlap?"
    assert at.session_state["messages"][1]["searched"] == ""  # the first question was searched as typed


def test_clear_deletes_the_vectors_and_resets_the_page(pinecone_client):
    at = load_sample(add_keys(start()))
    at = ask(at, "What is a chunk?")
    at.button(key="clear").click().run()
    assert not at.exception
    assert at.session_state["doc"] is None and at.session_state["messages"] == []
    assert at.session_state["uploader_n"] == 1
    assert pinecone_client.indexes["rag-document-qa"].namespaces == {}


def test_keys_from_secrets_work_without_typing_anything():
    at = start(BUILTIN)
    assert not any("Add your Cohere" in w.value for w in at.warning)
    assert any("has its own API keys" in i.value for i in at.sidebar.info)
    at = ask(load_sample(at), "What is a chunk?")
    assert not at.exception and at.session_state["questions"] == 1


def test_the_question_limit_applies_to_the_built_in_keys():
    at = load_sample(start({**BUILTIN, "RAG_MAX_QUESTIONS": 1}))
    at = ask(at, "What is a chunk?")
    at = ask(at, "What is overlap?")
    assert at.session_state["questions"] == 1
    assert any("allows 1 questions" in w.value for w in at.warning)


def test_the_question_limit_does_not_apply_to_keys_the_visitor_typed():
    at = start({**BUILTIN, "RAG_MAX_QUESTIONS": 1})
    at = load_sample(add_keys(at))
    at = ask(ask(at, "What is a chunk?"), "What is overlap?")
    assert at.session_state["questions"] == 2


def test_a_rejected_key_shows_a_plain_message(cohere_client):
    cohere_client.errors["embed"] = [sdk_error("cohere.errors", 401)]
    at = load_sample(add_keys(start()))
    assert not at.exception
    assert any("Cohere rejected the API key" in e.value for e in at.error)
    assert at.session_state["doc"] is None


def test_check_keys_reports_each_service():
    at = add_keys(start())
    at.button(key="check_keys").click().run()
    assert [s.value for s in at.success] == ["Cohere key works.", "Pinecone key works."]


def test_check_keys_explains_a_failure(cohere_client):
    cohere_client.errors["embed"] = [sdk_error("cohere.errors", 401)]
    at = add_keys(start())
    at.button(key="check_keys").click().run()
    assert any("Cohere rejected the API key" in e.value for e in at.error)
    assert [s.value for s in at.success] == ["Pinecone key works."]


def test_check_keys_without_keys_asks_for_them():
    at = start()
    at.button(key="check_keys").click().run()
    assert any("Enter both keys first" in w.value for w in at.warning)


def test_an_uploaded_pdf_is_read_and_clear_gives_a_fresh_uploader(monkeypatch, sample_bytes):
    def uploader(label, type=None, key=None, **kwargs):
        return FakeUpload("mine.pdf", sample_bytes) if key == "uploader_0" else None  # a new key is an empty box

    monkeypatch.setattr(st, "file_uploader", uploader)
    at = add_keys(start())
    assert at.session_state["doc"].filename == "mine.pdf"
    at.button(key="clear").click().run()
    assert at.session_state["doc"] is None  # it is not read again, because the uploader was replaced


def test_a_scanned_pdf_gets_a_clear_message(monkeypatch):
    blank = fitz.open()
    blank.new_page()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: FakeUpload("scan.pdf", blank.tobytes()))
    at = add_keys(start())
    assert any("probably a scan" in e.value for e in at.error)
