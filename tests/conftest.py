import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fakes import FakeCohereClient, FakePinecone  # noqa: E402

from rag.cohere_service import CohereService  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402

SAMPLE_PDF = ROOT / "samples" / "how-rag-works.pdf"


@pytest.fixture
def sample_bytes() -> bytes:
    return SAMPLE_PDF.read_bytes()


@pytest.fixture
def cohere_client() -> FakeCohereClient:
    return FakeCohereClient()


@pytest.fixture
def pinecone_client() -> FakePinecone:
    return FakePinecone()


@pytest.fixture
def settings() -> Settings:
    return Settings(namespace_ttl_hours=24)


@pytest.fixture
def clock():
    """A clock the tests can move. Calling it returns the current fake time; sleeping advances it."""

    class Clock:
        now = 1_800_000_000.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    return Clock()


@pytest.fixture
def pipeline(cohere_client, pinecone_client, settings, clock) -> RAGPipeline:
    cohere_service = CohereService("key", settings, client=cohere_client, sleep=clock.sleep)
    store = VectorStore("key", settings, client=pinecone_client, clock=clock, sleep=clock.sleep)
    return RAGPipeline(cohere_service, store, settings)
