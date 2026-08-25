"""Unit tests for eval metrics, golden loading, ablation, and report rendering."""

from __future__ import annotations

from pathlib import Path

from app.retrieval.models import Hit
from eval.ablation import (
    EmptySearcher,
    IdentityReranker,
    build_dense_only,
    build_hybrid,
    build_hybrid_rerank,
    evaluate_retriever,
    run_ablations,
)
from eval.golden import load_golden_set, normalize_arxiv_id
from eval.metrics import (
    aggregate,
    purity_at_k,
    recall_at_k,
    reciprocal_rank,
    score_case,
)
from eval.models import EvalReport, GoldenCase, RagasScores
from eval.report import ablation_table, render_markdown, write_report
from tests.retrieval_helpers import make_chunk


def test_normalize_arxiv_id_strips_version() -> None:
    assert normalize_arxiv_id("1706.03762v7") == "1706.03762"
    assert normalize_arxiv_id("1706.03762") == "1706.03762"


def test_load_default_golden_set() -> None:
    golden = load_golden_set()
    assert golden.version == 1
    assert len(golden.cases) >= 3
    assert all(c.question and c.reference_answer for c in golden.cases)


def test_recall_mrr_hit_helpers() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"c", "z"}
    assert recall_at_k(retrieved, relevant) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 1.0 / 3
    assert purity_at_k(retrieved, relevant) == 1.0 / 3
    assert recall_at_k([], {"a"}) == 0.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert purity_at_k([], {"a"}) == 0.0


def test_score_case_paper_level(fixture_corpus) -> None:
    attn, _bert, _cnn = fixture_corpus
    case = GoldenCase(
        id="attn",
        question="attention?",
        reference_answer="...",
        relevant_arxiv_ids=["attn"],
    )
    hits = [Hit(chunk=attn, score=1.0)]
    scored = score_case(case, hits, k=1)
    assert scored.hit is True
    assert scored.recall == 1.0
    assert scored.reciprocal_rank == 1.0


def test_score_case_chunk_level(fixture_corpus) -> None:
    attn, bert, _cnn = fixture_corpus
    case = GoldenCase(
        id="chunk",
        question="q",
        reference_answer="a",
        relevant_chunk_ids=["attn::0"],
    )
    hits = [Hit(chunk=bert, score=0.9), Hit(chunk=attn, score=0.5)]
    scored = score_case(case, hits, k=2)
    assert scored.hit is True
    assert scored.reciprocal_rank == 0.5


def test_aggregate_means() -> None:
    cases = [
        score_case(
            GoldenCase(
                id="1",
                question="q",
                reference_answer="a",
                relevant_chunk_ids=["attn::0"],
            ),
            [Hit(chunk=make_chunk("attn::0", "x"), score=1.0)],
            k=1,
        ),
        score_case(
            GoldenCase(
                id="2",
                question="q",
                reference_answer="a",
                relevant_chunk_ids=["missing::0"],
            ),
            [Hit(chunk=make_chunk("attn::0", "x"), score=1.0)],
            k=1,
        ),
    ]
    agg = aggregate(cases, k=1)
    assert agg.n == 2
    assert agg.hit_rate == 0.5
    assert agg.mrr == 0.5
    assert agg.purity_at_k == 0.5


class _FixedSearcher:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def search(self, query: str, top_k: int) -> list[Hit]:  # noqa: ARG002
        return self._hits[:top_k]


def test_ablation_variants_change_ranking(fixture_corpus) -> None:
    attn, bert, cnn = fixture_corpus
    # Dense prefers CNN; BM25-style sparse prefers attn for "attention".
    dense = _FixedSearcher(
        [
            Hit(chunk=cnn, score=0.9, dense_score=0.9),
            Hit(chunk=bert, score=0.5, dense_score=0.5),
            Hit(chunk=attn, score=0.1, dense_score=0.1),
        ]
    )
    sparse = _FixedSearcher(
        [
            Hit(chunk=attn, score=5.0, bm25_score=5.0),
            Hit(chunk=bert, score=1.0, bm25_score=1.0),
        ]
    )

    class PromoteAttnReranker:
        def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:  # noqa: ARG002
            promoted = sorted(
                hits,
                key=lambda h: (0 if h.chunk_id != "attn::0" else 1, h.score),
                reverse=True,
            )
            out: list[Hit] = []
            for i, hit in enumerate(promoted[:top_k]):
                score = float(len(promoted) - i)
                out.append(hit.model_copy(update={"score": score, "rerank_score": score}))
            return out

    cases = [
        GoldenCase(
            id="q1",
            question="attention",
            reference_answer="...",
            relevant_chunk_ids=["attn::0"],
        )
    ]
    dense_pipe = build_dense_only(dense, corpus_size=3, retrieval_top_k=3, rerank_top_k=2)
    hybrid_pipe = build_hybrid(sparse, dense, corpus_size=3, retrieval_top_k=3, rerank_top_k=2)
    full_pipe = build_hybrid_rerank(
        sparse,
        dense,
        PromoteAttnReranker(),
        corpus_size=3,
        retrieval_top_k=3,
        rerank_top_k=2,
    )

    dense_scores = evaluate_retriever(cases, dense_pipe.search, k=1)
    hybrid_scores = evaluate_retriever(cases, hybrid_pipe.search, k=1)
    full_scores = evaluate_retriever(cases, full_pipe.search, k=1)

    # Dense@1 misses attn; hybrid and rerank should hit.
    assert dense_scores.hit_rate == 0.0
    assert hybrid_scores.hit_rate == 1.0
    assert full_scores.hit_rate == 1.0
    assert full_pipe.search("attention").hits[0].chunk_id == "attn::0"


def test_run_ablations_returns_three_rows(fixture_corpus) -> None:
    attn, bert, cnn = fixture_corpus
    dense = _FixedSearcher([Hit(chunk=attn, score=0.9, dense_score=0.9)])
    rows = run_ablations(
        [
            GoldenCase(
                id="q",
                question="attention",
                reference_answer="a",
                relevant_chunk_ids=["attn::0"],
            )
        ],
        chunks=fixture_corpus,
        dense=dense,
        reranker=IdentityReranker(),
        retrieval_top_k=3,
        rerank_top_k=2,
        k=1,
    )
    assert [r.name for r in rows] == ["dense_only", "hybrid", "hybrid_rerank"]
    assert all(r.scores.n == 1 for r in rows)
    assert EmptySearcher().search("x", 3) == []
    _ = bert, cnn  # corpus fixture used via BM25 inside run_ablations


def test_report_markdown_and_write(tmp_path: Path) -> None:
    from eval.models import AblationRow, AggregatedScores, CaseRetrievalScores

    report = EvalReport(
        golden_path="eval/data/golden.json",
        corpus_size=22,
        ablations=[
            AblationRow(
                name="dense_only",
                description="baseline",
                scores=AggregatedScores(
                    hit_rate=0.5,
                    recall_at_k=0.5,
                    mrr=0.5,
                    n=2,
                    k=5,
                    cases=[
                        CaseRetrievalScores(
                            case_id="a",
                            hit=True,
                            recall=1.0,
                            reciprocal_rank=1.0,
                            top_ids=["1706.03762"],
                        )
                    ],
                ),
            )
        ],
        ragas=RagasScores(
            faithfulness=0.8,
            answer_relevancy=0.7,
            context_precision=0.9,
            n=2,
            skipped=False,
            detail="fake",
        ),
    )
    md = render_markdown(report)
    assert "dense_only" in md
    assert "Faithfulness" in md
    assert "0.500" in ablation_table(report.ablations)

    out = tmp_path / "latest.md"
    write_report(report, out)
    assert out.exists()
    assert out.with_suffix(".json").exists()
