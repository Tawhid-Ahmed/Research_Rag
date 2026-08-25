"""Token counting via tiktoken (same encoder family as chunking)."""

from __future__ import annotations

from functools import lru_cache

from app.llm.types import ChatMessage

DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=4)
def _encoding(name: str = DEFAULT_ENCODING):  # type: ignore[no-untyped-def]
    import tiktoken

    return tiktoken.get_encoding(name)


def count_text_tokens(text: str, *, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Return the number of tokens in ``text``."""

    if not text:
        return 0
    return len(_encoding(encoding_name).encode(text))


def count_message_tokens(
    messages: list[ChatMessage],
    *,
    encoding_name: str = DEFAULT_ENCODING,
) -> int:
    """Approximate chat-prompt token count (role + content, no special tokens)."""

    total = 0
    for message in messages:
        total += count_text_tokens(message.role, encoding_name=encoding_name)
        total += count_text_tokens(message.content, encoding_name=encoding_name)
        total += 4  # rough per-message overhead used by many OpenAI-style counters
    return total
