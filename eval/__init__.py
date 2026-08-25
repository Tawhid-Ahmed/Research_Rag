"""Evaluation harness: retrieval metrics + RAGAS answer quality and reports."""

from eval.golden import load_golden_set
from eval.metrics import aggregate, recall_at_k, reciprocal_rank, score_case
from eval.models import EvalReport, GoldenSet

__all__ = [
    "EvalReport",
    "GoldenSet",
    "aggregate",
    "load_golden_set",
    "recall_at_k",
    "reciprocal_rank",
    "score_case",
]
