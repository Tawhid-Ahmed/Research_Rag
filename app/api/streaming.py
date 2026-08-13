"""SSE event helpers for the query streaming endpoint.

Yields payloads suitable for ``sse_starlette.EventSourceResponse`` (dicts with
a ``data`` field). The library adds the ``data:`` framing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.llm.sse import DONE_SENTINEL
from app.llm.types import StreamChunk


async def iter_query_sse(
    *,
    meta: dict[str, Any],
    token_stream: AsyncIterator[StreamChunk],
) -> AsyncIterator[dict[str, str]]:
    """Yield metadata, then token frames, then ``[DONE]``."""

    yield {"data": json.dumps({"type": "meta", **meta}, ensure_ascii=False)}
    async for chunk in token_stream:
        if chunk.done:
            continue
        if chunk.text:
            yield {"data": json.dumps({"type": "token", "text": chunk.text}, ensure_ascii=False)}
    yield {"data": DONE_SENTINEL}
