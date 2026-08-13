"""Shared helpers for concrete LLM providers."""

from __future__ import annotations

from app.llm.types import ChatMessage


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat messages into a plain prompt for text-generation APIs."""

    parts = [f"{msg.role}: {msg.content}" for msg in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def split_system_and_chat(
    messages: list[ChatMessage],
) -> tuple[str | None, list[ChatMessage]]:
    """Pull leading system messages into one string; leave the rest as chat."""

    system_parts: list[str] = []
    chat: list[ChatMessage] = []
    for msg in messages:
        if msg.role == "system" and not chat:
            system_parts.append(msg.content)
        else:
            chat.append(msg)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, chat
