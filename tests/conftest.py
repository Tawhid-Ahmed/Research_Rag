"""Shared fixtures for retrieval tests: a tiny labeled corpus."""

from __future__ import annotations

import pytest

from app.ingest.models import Chunk
from tests.retrieval_helpers import make_chunk


@pytest.fixture(autouse=True)
def _reset_sse_starlette_appstatus() -> None:
    """Avoid cross-test event-loop binding errors from sse-starlette."""

    try:
        from sse_starlette.sse import AppStatus
    except ImportError:
        return
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None


@pytest.fixture
def fixture_corpus() -> list[Chunk]:
    """Three chunks with obvious lexical + topical differences."""

    return [
        make_chunk(
            "attn::0",
            "Multi-head self-attention lets the transformer attend to different representation subspaces.",
            title="Attention Is All You Need",
        ),
        make_chunk(
            "bert::0",
            "BERT pretrains bidirectional transformers with masked language modeling.",
            title="BERT",
        ),
        make_chunk(
            "cnn::0",
            "ResNet residual connections train very deep convolutional networks on ImageNet photos.",
            title="ResNet",
        ),
    ]
