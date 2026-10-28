"""The Cohere calls: embeddings, reranking, the grounded answer and the follow-up rewrite."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cohere

from .config import Settings
from .errors import status_of
from .prompts import ANSWER_PREAMBLE, REWRITE_PREAMBLE
from .vector_store import Hit

EMBED_BATCH = 96  # Cohere takes at most 96 texts in one embed call
RETRIES = 3


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    cited_ids: frozenset[str]


class CohereService:
    def __init__(
        self,
        api_key: str,
        settings: Settings,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = client if client is not None else cohere.Client(api_key)
        self._s = settings
        self._sleep = sleep

    def _call(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
        """Call the API and, when the key is being rate limited, wait and try again a few times."""
        delay = 2.0
        for attempt in range(RETRIES + 1):
            try:
                return fn(**kwargs)
            except Exception as exc:
                if status_of(exc) != 429 or attempt == RETRIES:
                    raise
                self._sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def check(self) -> None:
        """Cheapest call that proves the key works."""
        self.embed_query("ping")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            response = self._call(
                self._client.embed,
                texts=list(texts[start:start + EMBED_BATCH]),
                model=self._s.embed_model,
                input_type="search_document",
            )
            vectors.extend([float(x) for x in vector] for vector in response.embeddings)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._call(self._client.embed, texts=[text], model=self._s.embed_model, input_type="search_query")
        return [float(x) for x in response.embeddings[0]]

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        """Return (position in `documents`, relevance score) for the best top_n, best first."""
        response = self._call(
            self._client.rerank,
            query=query,
            documents=list(documents),
            top_n=top_n,
            model=self._s.rerank_model,
        )
        return [(int(r.index), float(r.relevance_score)) for r in response.results]

    def answer(self, question: str, hits: Sequence[Hit]) -> GeneratedAnswer:
        documents = [{"id": h.chunk_id, "title": f"Page {h.page_start}", "snippet": h.text} for h in hits]
        response = self._call(
            self._client.chat,
            message=question,
            model=self._s.chat_model,
            preamble=ANSWER_PREAMBLE,
            documents=documents,
            temperature=self._s.temperature,
        )
        cited: set[str] = set()
        for citation in getattr(response, "citations", None) or []:
            cited.update(str(i) for i in getattr(citation, "document_ids", None) or [])
        return GeneratedAnswer(text=(getattr(response, "text", "") or "").strip(), cited_ids=frozenset(cited))

    def standalone_question(self, question: str, history: Sequence[tuple[str, str]], max_turns: int = 3) -> str:
        """Rewrite a follow-up so it makes sense without the conversation. Falls back to the original question."""
        recent = history[-2 * max_turns:]
        lines = [f"{'User' if role == 'user' else 'Assistant'}: {content[:300]}" for role, content in recent]
        prompt = "Conversation:\n" + "\n".join(lines) + f"\n\nFollow-up question: {question}\n\nStandalone question:"
        response = self._call(
            self._client.chat,
            message=prompt,
            model=self._s.chat_model,
            preamble=REWRITE_PREAMBLE,
            temperature=0.0,
        )
        rewritten = " ".join((getattr(response, "text", "") or "").split())
        if not rewritten or len(rewritten) > 3 * len(question) + 200:
            return question
        return rewritten
