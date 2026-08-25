"""Tests for shared LLM provider helpers."""

from __future__ import annotations

from app.llm.providers.base import messages_to_prompt, split_system_and_chat
from app.llm.types import ChatMessage


def test_messages_to_prompt_includes_assistant_prefix() -> None:
    messages = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="Hi"),
    ]
    prompt = messages_to_prompt(messages)
    assert "system: Be brief." in prompt
    assert "user: Hi" in prompt
    assert prompt.endswith("assistant:")


def test_split_system_and_chat() -> None:
    messages = [
        ChatMessage(role="system", content="A"),
        ChatMessage(role="system", content="B"),
        ChatMessage(role="user", content="Q"),
        ChatMessage(role="assistant", content="Ans"),
    ]
    system, chat = split_system_and_chat(messages)
    assert system == "A\n\nB"
    assert [m.role for m in chat] == ["user", "assistant"]
