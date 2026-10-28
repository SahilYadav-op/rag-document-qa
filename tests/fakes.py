"""In-memory stand-ins for Cohere and Pinecone, so the tests run offline.

The fake embedding is a hashed bag of words, so texts that share words get similar vectors. That lets the tests check that
retrieval really finds the right chunk, not only that the calls were made.
"""
from __future__ import annotations

import hashlib
import math
import re
from types import SimpleNamespace

DIM = 1024


def vector_of(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) < 3:
            continue
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _overlap(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(wa & wb) / (len(wa | wb) or 1)


class ApiError(Exception):
    """Looks like the SDK exceptions: carries status_code."""

    def __init__(self, status_code: int, message: str = "error"):
        super().__init__(message)
        self.status_code = status_code


def sdk_error(module: str, status: int) -> ApiError:
    """An error that looks like one from the given SDK module (the module name decides whose error it is)."""
    return type("SdkError", (ApiError,), {"__module__": module})(status)


class FakeCohereClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.errors: dict[str, list[Exception]] = {}  # method name -> exceptions to raise before succeeding

    def _enter(self, method: str, **kwargs):
        self.calls.append((method, kwargs))
        queue = self.errors.get(method)
        if queue:
            raise queue.pop(0)

    def embed(self, texts, model, input_type):
        self._enter("embed", texts=texts, model=model, input_type=input_type)
        return SimpleNamespace(embeddings=[vector_of(t) for t in texts])

    def rerank(self, query, documents, top_n, model):
        self._enter("rerank", query=query, documents=documents, top_n=top_n, model=model)
        scored = sorted(((i, _overlap(query, d)) for i, d in enumerate(documents)), key=lambda p: -p[1])[:top_n]
        return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in scored])

    def chat(self, message, model, preamble, temperature, documents=None):
        self._enter("chat", message=message, model=model, preamble=preamble, documents=documents, temperature=temperature)
        if documents is None:  # the follow-up rewrite
            question = message.split("Follow-up question:")[-1].split("Standalone question:")[0].strip()
            return SimpleNamespace(text=f"REWRITTEN {question}", citations=None)
        first = documents[0]
        return SimpleNamespace(
            text=f"According to the document: {first['snippet'][:80]}",
            citations=[SimpleNamespace(document_ids=[first["id"]])],
        )

    def calls_to(self, method: str) -> list[dict]:
        return [kwargs for name, kwargs in self.calls if name == method]


class FakeIndex:
    def __init__(self):
        self.namespaces: dict[str, dict[str, tuple[list[float], dict]]] = {}
        self.upsert_calls = 0

    def upsert(self, vectors, namespace="", batch_size=None, show_progress=True):
        self.upsert_calls += 1
        store = self.namespaces.setdefault(namespace, {})
        for record in vectors:
            store[record["id"]] = (record["values"], record["metadata"])

    def query(self, vector, top_k, namespace="", include_metadata=False):
        rows = self.namespaces.get(namespace, {})
        scored = sorted(
            ((sum(a * b for a, b in zip(vector, vec)), rid, meta) for rid, (vec, meta) in rows.items()),
            key=lambda row: -row[0],
        )[:top_k]
        return SimpleNamespace(matches=[SimpleNamespace(id=rid, score=score, metadata=meta) for score, rid, meta in scored])

    def delete(self, delete_all=False, namespace="", ids=None):
        if namespace not in self.namespaces:
            raise ApiError(404, "namespace not found")
        del self.namespaces[namespace]

    def describe_index_stats(self):
        return SimpleNamespace(
            namespaces={name: SimpleNamespace(vector_count=len(rows)) for name, rows in self.namespaces.items()}
        )


class FakePinecone:
    def __init__(self, dimension: int = DIM):
        self.indexes: dict[str, FakeIndex] = {}
        self.dimension = dimension
        self.created: list[dict] = []
        self.fail_create_with: Exception | None = None

    def list_indexes(self):
        return [SimpleNamespace(name=n) for n in self.indexes]

    def has_index(self, name):
        return name in self.indexes

    def create_index(self, name, dimension, metric, spec):
        if self.fail_create_with:
            raise self.fail_create_with
        self.created.append({"name": name, "dimension": dimension, "metric": metric, "cloud": spec.cloud, "region": spec.region})
        self.indexes[name] = FakeIndex()

    def describe_index(self, name):
        return SimpleNamespace(status=SimpleNamespace(ready=True), dimension=self.dimension)

    def Index(self, name):  # noqa: N802
        return self.indexes[name]

    def index(self, name: str) -> FakeIndex:
        return self.indexes[name]


class FakeUpload:
    """What st.file_uploader returns."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data
