"""Streamlit app: upload a PDF, ask questions, and see which pages each answer came from."""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import streamlit as st

from rag.config import settings_from_env
from rag.errors import explain
from rag.keys import load_keys
from rag.pipeline import Answer, build_pipeline, file_digest
from rag.render import neutralize_links

ROOT = Path(__file__).resolve().parent
SAMPLE_PDF = ROOT / "samples" / "how-rag-works.pdf"
REPO_URL = "https://github.com/SahilYadav-op/rag-document-qa"
GITHUB_URL = "https://github.com/SahilYadav-op"
LINKEDIN_URL = "https://www.linkedin.com/in/sahilgod01/"

st.set_page_config(page_title="RAG Document QA", layout="centered")


def _secrets() -> dict:
    """Top-level values from Streamlit secrets. Empty when the app has no secrets file."""
    try:
        return {k: v for k, v in st.secrets.items() if isinstance(v, (str, int, float, bool))}
    except Exception:
        return {}


def _init_state() -> None:
    for name, default in (
        ("uploader_n", 0),
        ("doc", None),
        ("messages", []),
        ("questions", 0),
        ("pipeline", None),
        ("pipeline_fp", None),
    ):
        st.session_state.setdefault(name, default)


def _reset() -> None:
    """Button callback: delete the document's vectors and return to the empty state."""
    pipeline, doc = st.session_state.get("pipeline"), st.session_state.get("doc")
    if pipeline is not None and doc is not None:
        pipeline.clear(doc.namespace)
    st.session_state.doc = None
    st.session_state.messages = []
    st.session_state.questions = 0
    st.session_state.uploader_n += 1  # a new key gives a fresh, empty file box


def _pages_label(start: int, end: int) -> str:
    return f"Page {start}" if start == end else f"Pages {start} to {end}"


def _show_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            cited = ", cited in the answer" if source.cited else ""
            st.markdown(f"**{_pages_label(source.page_start, source.page_end)}** (relevance {source.score:.2f}{cited})")
            st.text(source.text)  # plain text, so nothing inside the document is rendered as markdown


def _show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("searched"):
            st.caption(neutralize_links(f"Searched for: {message['searched']}"))
        st.markdown(neutralize_links(message["content"]))
        _show_sources(message.get("sources") or [])


_init_state()
secrets = _secrets()
settings_base = settings_from_env({**os.environ, **secrets})
builtin_cohere, builtin_pinecone = load_keys({**os.environ, **secrets})
has_builtin_keys = bool(builtin_cohere and builtin_pinecone)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    if has_builtin_keys:
        st.info("This app has its own API keys. Paste yours below if you would rather use your own quota.")
    st.text_input("Cohere API key", type="password", key="cohere_key")
    st.text_input("Pinecone API key", type="password", key="pinecone_key")
    check_clicked = st.button("Check keys", key="check_keys")
    check_area = st.container()  # the result of "Check keys" is written here, under the button
    st.caption("Keys stay in this browser session. Nothing is written to disk.")

    with st.expander("Retrieval settings"):
        st.caption("Chunk settings apply to the next document you upload.")
        chunk_size = st.slider("Chunk size (characters)", 400, 2000, settings_base.chunk_size, 100, key="chunk_size")
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, settings_base.chunk_overlap, 25, key="chunk_overlap")
        top_k = st.slider("Chunks to search", 3, 20, settings_base.retrieve_top_k, key="top_k")
        top_n = st.slider("Chunks sent to the model", 1, 8, settings_base.rerank_top_n, key="top_n")
        min_relevance = st.slider("Ignore matches scoring below", 0.0, 0.9, settings_base.min_relevance, 0.05, key="min_relevance")
        rewrite = st.checkbox(
            "Understand follow-up questions (one extra model call)", value=settings_base.rewrite_followups, key="rewrite"
        )

    st.button("Clear and start over", key="clear", on_click=_reset)
    with st.expander("How it works"):
        st.markdown(
            "1. The PDF is read page by page and split into overlapping chunks.\n"
            "2. Cohere turns each chunk into an embedding and Pinecone stores it.\n"
            "3. Your question is embedded and the nearest chunks are fetched.\n"
            "4. A reranker keeps the best few and a language model answers from them only.\n"
            "5. The answer shows the pages it used."
        )
    st.markdown(f"[Source code]({REPO_URL})")
    st.divider()
    st.markdown(f"Built by **Sahil Yadav**  \n[GitHub]({GITHUB_URL})  \n[LinkedIn]({LINKEDIN_URL})")

chunk_overlap = min(chunk_overlap, chunk_size // 2)
settings = replace(
    settings_base,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    retrieve_top_k=top_k,
    rerank_top_n=min(top_n, top_k),
    min_relevance=min_relevance,
    rewrite_followups=rewrite,
)

typed_cohere = st.session_state.get("cohere_key", "").strip()
typed_pinecone = st.session_state.get("pinecone_key", "").strip()
cohere_key = typed_cohere or builtin_cohere or ""
pinecone_key = typed_pinecone or builtin_pinecone or ""
uses_builtin_keys = (not typed_cohere and bool(builtin_cohere)) or (not typed_pinecone and bool(builtin_pinecone))

pipeline = None
if cohere_key and pinecone_key:
    fingerprint = hashlib.sha256(f"{cohere_key}|{pinecone_key}|{settings}".encode()).hexdigest()
    if st.session_state.pipeline_fp != fingerprint:
        st.session_state.pipeline = build_pipeline(cohere_key, pinecone_key, settings)
        st.session_state.pipeline_fp = fingerprint
    pipeline = st.session_state.pipeline

if check_clicked:
    with check_area:
        if pipeline is None:
            st.warning("Enter both keys first.")
        else:
            for label, test in (("Cohere", pipeline.cohere.check), ("Pinecone", pipeline.store.check)):
                try:
                    test()
                    st.success(f"{label} key works.")
                except Exception as exc:
                    st.error(explain(exc))

# ---------------------------------------------------------------- main page
st.title("RAG Document QA")
st.caption("Upload a PDF and ask questions about it. Answers come only from the document and show the pages they used.")

if pipeline is None:
    st.warning("Add your Cohere and Pinecone API keys in the sidebar to begin.")

uploaded = st.file_uploader("Upload a PDF", type="pdf", key=f"uploader_{st.session_state.uploader_n}")
sample_slot = st.empty()  # holds the sample button, and is emptied once a document is loaded
use_sample = sample_slot.button("Use the sample PDF", key="use_sample") if st.session_state.doc is None else False

source = None
if uploaded is not None:
    source = (uploaded.name, uploaded.getvalue())
elif use_sample:
    source = (SAMPLE_PDF.name, SAMPLE_PDF.read_bytes())

if source is not None and pipeline is not None:
    filename, data = source
    current = st.session_state.doc
    if current is None or current.digest != file_digest(data):
        if current is not None:
            pipeline.clear(current.namespace)
            st.session_state.doc, st.session_state.messages, st.session_state.questions = None, [], 0
        problem = None
        with st.status("Reading the document", expanded=True) as status:
            try:
                st.session_state.doc = pipeline.ingest(data, filename, progress=status.write)
            except Exception as exc:
                problem = explain(exc)
                status.update(label="Could not process the document", state="error", expanded=True)
            else:
                doc = st.session_state.doc
                status.update(label=f"Ready: {doc.pages} pages, {doc.chunk_count} chunks", state="complete", expanded=False)
                sample_slot.empty()
        if problem:  # shown outside the status box, which collapses when it finishes
            st.error(problem)

doc = st.session_state.doc
if doc is not None:
    st.caption(f"{doc.filename}: {doc.pages} pages, {doc.chunk_count} chunks")
elif pipeline is not None:
    st.info("Upload a PDF, or use the sample, to begin.")

for message in st.session_state.messages:
    _show_message(message)

question = st.chat_input("Ask a question about the document", key="question", disabled=doc is None)
if question and doc is not None and pipeline is not None:
    limit = settings.max_questions
    if limit and uses_builtin_keys and st.session_state.questions >= limit:
        st.warning(f"This demo allows {limit} questions per session on its own keys. Paste your own keys to continue.")
    else:
        history = [(m["role"], m["content"]) for m in st.session_state.messages]
        user_message = {"role": "user", "content": question}
        _show_message(user_message)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document"):
                    answer: Answer = pipeline.ask(doc.namespace, question, history)
            except Exception as exc:
                st.error(explain(exc))
            else:
                reply = {
                    "role": "assistant",
                    "content": answer.text,
                    "sources": answer.sources,
                    "searched": answer.searched if answer.searched != question else "",
                }
                if reply["searched"]:
                    st.caption(neutralize_links(f"Searched for: {reply['searched']}"))
                st.markdown(neutralize_links(reply["content"]))
                _show_sources(reply["sources"])
                st.session_state.messages += [user_message, reply]
                st.session_state.questions += 1
