"""Pipeline settings.

The defaults are the values the original week-7 project used. Every field can be overridden from the
environment (or from Streamlit secrets), see ENV_NAMES.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # retrieval
    retrieve_top_k: int = 10
    rerank_top_n: int = 3
    min_relevance: float = 0.0
    rewrite_followups: bool = True

    # models
    embed_model: str = "embed-english-v3.0"
    rerank_model: str = "rerank-english-v3.0"
    chat_model: str = "command-r-plus-08-2024"
    temperature: float = 0.3

    # pinecone
    index_name: str = "rag-document-qa"
    index_cloud: str = "aws"
    index_region: str = "us-east-1"
    embed_dim: int = 1024

    # limits (they protect your API quota when the app runs on your own keys)
    max_pages: int = 80
    max_chunks: int = 600
    max_questions: int = 0  # 0 means no limit; only applies when the app uses keys from secrets or the environment
    namespace_ttl_hours: int = 24

    def __post_init__(self) -> None:
        if self.chunk_size < 200:
            raise ValueError("chunk_size must be at least 200 characters")
        if not 0 <= self.chunk_overlap <= self.chunk_size // 2:
            raise ValueError("chunk_overlap must be between 0 and half of chunk_size")
        if self.rerank_top_n < 1 or self.retrieve_top_k < self.rerank_top_n:
            raise ValueError("rerank_top_n must be at least 1 and no larger than retrieve_top_k")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# field name -> (environment variable, converter)
ENV_NAMES = {
    "chunk_size": ("RAG_CHUNK_SIZE", int),
    "chunk_overlap": ("RAG_CHUNK_OVERLAP", int),
    "retrieve_top_k": ("RAG_RETRIEVE_TOP_K", int),
    "rerank_top_n": ("RAG_RERANK_TOP_N", int),
    "min_relevance": ("RAG_MIN_RELEVANCE", float),
    "rewrite_followups": ("RAG_REWRITE_FOLLOWUPS", _as_bool),
    "embed_model": ("COHERE_EMBED_MODEL", str),
    "rerank_model": ("COHERE_RERANK_MODEL", str),
    "chat_model": ("COHERE_CHAT_MODEL", str),
    "index_name": ("PINECONE_INDEX_NAME", str),
    "index_cloud": ("PINECONE_CLOUD", str),
    "index_region": ("PINECONE_REGION", str),
    "max_pages": ("RAG_MAX_PAGES", int),
    "max_chunks": ("RAG_MAX_CHUNKS", int),
    "max_questions": ("RAG_MAX_QUESTIONS", int),
    "namespace_ttl_hours": ("RAG_NAMESPACE_TTL_HOURS", int),
}


def settings_from_env(env: Mapping[str, object] | None = None) -> Settings:
    """Build Settings from environment-style variables. Unset or empty variables keep the default."""
    env = os.environ if env is None else env
    values = {}
    for field, (name, convert) in ENV_NAMES.items():
        raw = env.get(name)
        if raw is None or raw == "":
            continue
        values[field] = convert(raw)
    return Settings(**values)
