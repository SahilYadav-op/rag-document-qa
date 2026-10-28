"""The Pinecone calls: one shared index, and one namespace for every uploaded document."""
from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .chunking import Chunk
from .config import Settings
from .errors import ServiceError, status_of

UPSERT_BATCH = 100
_NAMESPACE = re.compile(r"^s(\d{9,11})-[0-9a-f]{6}$")  # s<unix time>-<random>, so old ones can be found and removed


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    score: float


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Pinecone response, whether the SDK returns an object or a dict."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class VectorStore:
    def __init__(
        self,
        api_key: str,
        settings: Settings,
        client: Any = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._pc = client if client is not None else Pinecone(api_key=api_key)
        self._s = settings
        self._clock = clock
        self._sleep = sleep
        self._index: Any = None

    def check(self) -> None:
        """Cheapest call that proves the key works."""
        self._pc.list_indexes()

    def new_namespace(self) -> str:
        return f"s{int(self._clock())}-{uuid.uuid4().hex[:6]}"

    def ensure_index(self) -> Any:
        """Create the index if it does not exist yet, wait until it is ready, and return a handle to it."""
        if self._index is not None:
            return self._index
        name = self._s.index_name
        if not self._pc.has_index(name):
            try:
                self._pc.create_index(
                    name=name,
                    dimension=self._s.embed_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self._s.index_cloud, region=self._s.index_region),
                )
            except Exception as exc:
                if status_of(exc) != 409:  # 409 means another session created it a moment ago, which is fine
                    raise
        deadline = self._clock() + 120
        while True:
            description = self._pc.describe_index(name)
            if _get(_get(description, "status"), "ready"):
                break
            if self._clock() > deadline:
                raise ServiceError("The Pinecone index is still not ready after two minutes. Try again shortly.")
            self._sleep(2)
        dimension = _get(description, "dimension")
        if dimension is not None and dimension != self._s.embed_dim:
            raise ServiceError(
                f"The Pinecone index '{name}' already exists with dimension {dimension}, but this app needs "
                f"{self._s.embed_dim}. Set PINECONE_INDEX_NAME to a new name."
            )
        self._index = self._pc.Index(name)
        return self._index

    def upsert(self, namespace: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], filename: str) -> None:
        index = self.ensure_index()
        records = [
            {
                "id": chunk.id,
                "values": list(vector),
                "metadata": {
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "filename": filename,
                },
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        index.upsert(vectors=records, namespace=namespace, batch_size=UPSERT_BATCH, show_progress=False)

    def wait_until_visible(self, namespace: str, expected: int, timeout: float = 20.0) -> bool:
        """A new serverless index can take a few seconds before queries see the vectors. Returns False on timeout."""
        index = self.ensure_index()
        deadline = self._clock() + timeout
        while True:
            namespaces = _get(index.describe_index_stats(), "namespaces") or {}
            count = _get(namespaces.get(namespace), "vector_count", 0) or 0
            if count >= expected:
                return True
            if self._clock() > deadline:
                return False
            self._sleep(1)

    def query(self, namespace: str, vector: Sequence[float], top_k: int) -> list[Hit]:
        index = self.ensure_index()
        response = index.query(vector=list(vector), top_k=top_k, namespace=namespace, include_metadata=True)
        hits = []
        for match in _get(response, "matches") or []:
            metadata = _get(match, "metadata") or {}
            hits.append(
                Hit(
                    chunk_id=str(_get(match, "id")),
                    text=str(metadata.get("text", "")),
                    page_start=int(metadata.get("page_start", 0)),
                    page_end=int(metadata.get("page_end", 0)),
                    score=float(_get(match, "score", 0.0)),
                )
            )
        return hits

    def delete_namespace(self, namespace: str) -> None:
        index = self.ensure_index()
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception as exc:
            if status_of(exc) != 404:  # the namespace is already gone
                raise

    def purge_stale(self, max_age_hours: float) -> int:
        """Delete this app's namespaces that are older than max_age_hours. Namespaces from other tools are left alone."""
        index = self.ensure_index()
        cutoff = self._clock() - max_age_hours * 3600
        removed = 0
        for name in list(_get(index.describe_index_stats(), "namespaces") or {}):
            match = _NAMESPACE.match(name)
            if match and int(match.group(1)) < cutoff:
                self.delete_namespace(name)
                removed += 1
        return removed
