"""Load and normalize the golden Q/A set."""

from __future__ import annotations

import json
import re
from pathlib import Path

from eval.models import GoldenCase, GoldenSet

DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden.json"

_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Strip version suffixes so ``1706.03762v7`` matches ``1706.03762``."""

    cleaned = arxiv_id.strip()
    return _VERSION_SUFFIX.sub("", cleaned)


def load_golden_set(path: Path | str | None = None) -> GoldenSet:
    """Parse a golden JSON file into a :class:`GoldenSet`."""

    golden_path = Path(path) if path is not None else DEFAULT_GOLDEN_PATH
    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    return GoldenSet.model_validate(raw)


def relevant_ids_for_case(case: GoldenCase) -> set[str]:
    """Return the labeled relevant IDs (chunk ids preferred, else arXiv ids)."""

    if case.relevant_chunk_ids:
        return set(case.relevant_chunk_ids)
    return {normalize_arxiv_id(item) for item in case.relevant_arxiv_ids}
