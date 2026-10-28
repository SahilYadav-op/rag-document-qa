"""Measure retrieval and answer quality on your own questions.

    python -m eval.run_eval --pdf samples/how-rag-works.pdf --questions eval/questions.sample.json

It needs real keys (COHERE_API_KEY and PINECONE_API_KEY in the environment, or in .streamlit/secrets.toml). The PDF is
stored in a temporary namespace, which is deleted at the end.

Each question in the JSON file has:
    question         what to ask
    expect           text that must appear in the passage that answers it (case does not matter)
    page             optional: the page that passage is on
    answer_contains  optional: text the final answer must contain

Measured per question:
    retrieved   the expected text is in the chunks returned by vector search (top-k)
    reranked    the expected text is in the chunks kept after reranking
    page        one of the kept chunks is on the expected page
    answered    the final answer contains `answer_contains`
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import settings_from_env  # noqa: E402
from rag.keys import SECRETS_FILE, load_keys, looks_like_placeholder  # noqa: E402
from rag.pipeline import RAGPipeline, build_pipeline  # noqa: E402


@dataclass(frozen=True)
class Item:
    question: str
    expect: str
    page: int | None = None
    answer_contains: str | None = None


@dataclass(frozen=True)
class Result:
    question: str
    retrieved: bool
    reranked: bool
    page: bool | None
    answered: bool | None
    answer: str


def load_items(path: Path) -> list[Item]:
    return [Item(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _has(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def evaluate(pipeline: RAGPipeline, namespace: str, items: list[Item]) -> list[Result]:
    k = pipeline.settings.retrieve_top_k
    results = []
    for item in items:
        hits = pipeline.store.query(namespace, pipeline.cohere.embed_query(item.question), k)
        answer = pipeline.ask(namespace, item.question)
        kept = answer.sources
        results.append(Result(
            question=item.question,
            retrieved=any(_has(h.text, item.expect) for h in hits),
            reranked=any(_has(s.text, item.expect) for s in kept),
            page=None if item.page is None else any(s.page_start <= item.page <= s.page_end for s in kept),
            answered=None if item.answer_contains is None else _has(answer.text, item.answer_contains),
            answer=answer.text,
        ))
    return results


def summarize(results: list[Result]) -> dict[str, str]:
    def rate(values):
        values = [v for v in values if v is not None]
        return f"{sum(values)}/{len(values)}" if values else "n/a"

    return {
        "retrieved": rate(r.retrieved for r in results),
        "reranked": rate(r.reranked for r in results),
        "page": rate(r.page for r in results),
        "answered": rate(r.answered for r in results),
    }


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "NO")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=ROOT / "samples" / "how-rag-works.pdf")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.sample.json")
    parser.add_argument("--top-k", type=int, help="chunks fetched by vector search (default from settings)")
    parser.add_argument("--top-n", type=int, help="chunks kept after reranking (default from settings)")
    args = parser.parse_args()

    cohere_key, pinecone_key = load_keys(secrets_file=SECRETS_FILE)
    if looks_like_placeholder(cohere_key) or looks_like_placeholder(pinecone_key):
        print("Set COHERE_API_KEY and PINECONE_API_KEY (environment or .streamlit/secrets.toml) to real keys first.")
        return 2

    settings = settings_from_env()
    top_k = args.top_k or settings.retrieve_top_k
    settings = replace(settings, retrieve_top_k=top_k, rerank_top_n=min(args.top_n or settings.rerank_top_n, top_k))
    pipeline = build_pipeline(cohere_key, pinecone_key, settings)
    doc = pipeline.ingest(args.pdf.read_bytes(), args.pdf.name, progress=print)
    try:
        results = evaluate(pipeline, doc.namespace, load_items(args.questions))
    finally:
        pipeline.clear(doc.namespace)

    print(f"\n{'retrieved':10}{'reranked':10}{'page':6}{'answered':10}question")
    for r in results:
        print(f"{_mark(r.retrieved):10}{_mark(r.reranked):10}{_mark(r.page):6}{_mark(r.answered):10}{r.question}")
    print("\nsummary:", json.dumps(summarize(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
