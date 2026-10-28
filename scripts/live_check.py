"""Check that your Cohere and Pinecone keys work with this app's exact calls. Run it before you deploy.

    python scripts/live_check.py

Keys come from the environment (COHERE_API_KEY, PINECONE_API_KEY) or from .streamlit/secrets.toml. It stores the sample PDF in a
temporary Pinecone namespace, asks one question, and deletes the namespace. The first run creates the index, which can take a
minute. It never prints a key.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import settings_from_env  # noqa: E402
from rag.errors import explain  # noqa: E402
from rag.keys import SECRETS_FILE, load_keys, looks_like_placeholder  # noqa: E402
from rag.pipeline import build_pipeline  # noqa: E402

SAMPLE = ROOT / "samples" / "how-rag-works.pdf"
QUESTION = "How many numbers are in an embedding?"


def main() -> int:
    cohere_key, pinecone_key = load_keys(secrets_file=SECRETS_FILE)
    if looks_like_placeholder(cohere_key) or looks_like_placeholder(pinecone_key):
        print("Set COHERE_API_KEY and PINECONE_API_KEY (environment or .streamlit/secrets.toml) to real keys first.")
        return 2

    settings = settings_from_env()
    pipeline = build_pipeline(cohere_key, pinecone_key, settings)
    results: list[tuple[str, bool, str]] = []

    def step(name, fn):
        started = time.time()
        try:
            note = fn()
            results.append((name, True, f"{time.time() - started:.1f}s  {note or ''}".strip()))
            return True
        except Exception as exc:  # report every failure in plain words
            results.append((name, False, explain(exc)))
            return False

    state = {}
    ok = (
        step("Cohere key and embedding model", lambda: pipeline.cohere.check())
        and step("Pinecone key", lambda: pipeline.store.check())
        and step("Index, embed and store the sample PDF", lambda: state.update(
            doc=pipeline.ingest(SAMPLE.read_bytes(), SAMPLE.name)) or f"{state['doc'].chunk_count} chunks")
        and step("Search, rerank and answer", lambda: state.update(
            answer=pipeline.ask(state["doc"].namespace, QUESTION)) or f"answer: {state['answer'].text[:90]!r}")
    )
    if "doc" in state:
        step("Delete the temporary namespace", lambda: pipeline.clear(state["doc"].namespace))

    print()
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name:42}  {detail}")
    if ok and "answer" in state:
        sources = [f"page {s.page_start}" for s in state["answer"].sources]
        print(f"\nsources used: {', '.join(sources) or 'none'}")
    print("\nRESULT:", "everything the app calls works" if ok else "a call failed; see the FAIL line above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
