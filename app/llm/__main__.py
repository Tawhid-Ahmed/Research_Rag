"""Command-line smoke test for the configured LLM provider.

Examples::

    python -m app.llm --prompt "Say hello in one sentence."
    python -m app.llm --prompt "Explain BM25 briefly." --stream
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app.config import get_settings
from app.llm.factory import build_provider
from app.llm.sse import iter_sse
from app.llm.types import ChatMessage, GenerationConfig


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.llm",
        description="Smoke-test the configured LLM provider.",
    )
    parser.add_argument("--prompt", required=True, help="user prompt to send")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream tokens (prints SSE-encoded lines when --sse is set)",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="with --stream, print Server-Sent Event lines instead of raw text",
    )
    return parser


async def _run(prompt: str, *, stream: bool, sse: bool) -> int:
    settings = get_settings()
    provider = build_provider(settings)
    messages = [ChatMessage(role="user", content=prompt)]
    config = GenerationConfig(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    print(f"provider={settings.llm_provider.value} model={settings.llm_model}")
    if not stream:
        text = await provider.generate(messages, config)
        print(text)
        return 0

    if sse:
        async for line in iter_sse(provider.stream_generate(messages, config)):
            print(line, end="")
        return 0

    async for chunk in provider.stream_generate(messages, config):
        if chunk.done:
            print()
            break
        print(chunk.text, end="", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args.prompt, stream=args.stream, sse=args.sse))


if __name__ == "__main__":
    sys.exit(main())
