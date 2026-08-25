"""SSE helpers that turn :class:`StreamChunk` iterators into event payloads.

Step 5 wires these into FastAPI with ``sse-starlette``. Keeping the encoding
here means the API route stays thin and the LLM package owns the wire format.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.llm.types import StreamChunk

DONE_SENTINEL = "[DONE]"


def encode_sse_chunk(chunk: StreamChunk) -> str:
    """Encode one chunk as an SSE ``data:`` line (including trailing blank line)."""

    if chunk.done:
        return f"data: {DONE_SENTINEL}\n\n"
    payload = json.dumps({"text": chunk.text}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def iter_sse(chunks: AsyncIterator[StreamChunk]) -> AsyncIterator[str]:
    """Yield SSE-encoded strings from a provider stream.

    Always ends with ``data: [DONE]`` even if the provider forgot ``done=True``.
    """

    saw_done = False
    async for chunk in chunks:
        if chunk.done:
            saw_done = True
        yield encode_sse_chunk(chunk)
    if not saw_done:
        yield encode_sse_chunk(StreamChunk(done=True))
