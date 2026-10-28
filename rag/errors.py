"""Errors, and a function that turns a failure from Cohere or Pinecone into one sentence a user can act on."""
from __future__ import annotations

import re


class RagError(Exception):
    """An error whose message is safe to show to the user as it is."""


class PdfError(RagError):
    """The file cannot be used as a PDF."""


class NoTextError(PdfError):
    """The PDF has no selectable text (for example a scan)."""


class PdfTooLargeError(PdfError):
    """The PDF is over the page or chunk limit."""


class ServiceError(RagError):
    """A problem with the vector database or the model service that we can describe ourselves."""


_LONG_TOKEN = re.compile(r"[A-Za-z0-9_\-]{24,}")


def status_of(exc: BaseException) -> int | None:
    """The HTTP status carried by a Cohere or Pinecone exception, if it has one."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def scrub(text: str, limit: int = 160) -> str:
    """Shorten an error message and blank out anything that looks like a key."""
    return _LONG_TOKEN.sub("<hidden>", " ".join(str(text).split()))[:limit]


def explain(exc: BaseException) -> str:
    if isinstance(exc, RagError):
        return str(exc)
    service = {"cohere": "Cohere", "pinecone": "Pinecone"}.get(type(exc).__module__.split(".")[0], "The service")
    status = status_of(exc)
    if status == 401:
        return f"{service} rejected the API key. Check that you pasted the right key."
    if status == 403:
        if service == "Pinecone":
            return ("Pinecone refused the request. On a free plan this usually means there is no room for another index "
                    "or namespace, or the key has no permission to create one.")
        return f"{service} refused the request. Check that the key has access to the model."
    if status == 404:
        return f"{service} could not find the model or index that was asked for. Check the names in the settings."
    if status == 429:
        return (f"{service} is rate limiting this key. Free and trial plans have per-minute and monthly limits. "
                "Wait a minute and try again.")
    if status is not None and status >= 500:
        return f"{service} is having problems (HTTP {status}). Try again in a moment."
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return f"The request to {service} timed out. Try again."
    return f"Something went wrong ({type(exc).__name__}). {scrub(exc)}"
