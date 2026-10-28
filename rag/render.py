"""Make model output safe to show as markdown.

A document can try to steer the model (indirect prompt injection). One common goal is to get the answer to contain a
markdown image or link whose address carries data out of the page. Links and images are therefore shown as plain text.
"""
from __future__ import annotations

import re

_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_URL = re.compile(r"\b(https?|ftp)://", re.IGNORECASE)
_ZERO_WIDTH_SPACE = "​"


def neutralize_links(text: str) -> str:
    text = _LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    # the zero-width space stops the renderer from turning the address into a link or fetching it as an image
    return _URL.sub(lambda m: f"{m.group(1)}:{_ZERO_WIDTH_SPACE}//", text)
