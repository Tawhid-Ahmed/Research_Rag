"""Shared fixtures for retrieval tests: a tiny labeled corpus."""

from __future__ import annotations

import pytest

from app.ingest.models import Chunk
from tests.retrieval_helpers import make_chunk


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
