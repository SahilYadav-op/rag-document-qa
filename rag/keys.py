"""Find the Cohere and Pinecone keys in the environment or in a Streamlit secrets file."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"


def looks_like_placeholder(value: str | None) -> bool:
    """True for empty values and for the text in the example files ("your_key_here")."""
    if not value:
        return True
    lowered = value.lower()
    return len(value) < 20 or lowered.startswith("your") or lowered.endswith("_here") or "api_key" in lowered


def load_keys(env: Mapping[str, object] | None = None, secrets_file: Path | None = None) -> tuple[str | None, str | None]:
    """Return (cohere_key, pinecone_key). The environment wins over the secrets file. Missing keys are None."""
    env = os.environ if env is None else env
    cohere_key, pinecone_key = env.get("COHERE_API_KEY"), env.get("PINECONE_API_KEY")
    if secrets_file is not None and (not cohere_key or not pinecone_key) and secrets_file.is_file():
        try:
            import tomllib  # Python 3.11 and later

            data = tomllib.loads(secrets_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        cohere_key = cohere_key or data.get("COHERE_API_KEY")
        pinecone_key = pinecone_key or data.get("PINECONE_API_KEY")
    return (str(cohere_key) if cohere_key else None, str(pinecone_key) if pinecone_key else None)
