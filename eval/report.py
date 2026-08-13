"""Render eval results as markdown (before/after ablation + optional RAGAS)."""

from __future__ import annotations

from pathlib import Path

from eval.models import AblationRow, EvalReport, RagasScores


def ablation_table(rows: list[AblationRow]) -> str:
    """Markdown table comparing retrieval configurations."""

    lines = [
        "| Config | Hit-rate | Recall@k | MRR | Purity@k | n | k |",
        "|--------|---------:|---------:|----:|---------:|--:|--:|",
    ]
    for row in rows:
        s = row.scores
        lines.append(
            f"| `{row.name}` | {s.hit_rate:.3f} | {s.recall_at_k:.3f} | "
            f"{s.mrr:.3f} | {s.purity_at_k:.3f} | {s.n} | {s.k} |"
        )
    return "\n".join(lines)


def ragas_section(scores: RagasScores | None) -> str:
    if scores is None:
        return "_RAGAS not run (retrieval-only). Pass `--with-ragas` to enable._"
    if scores.skipped:
        return f"_RAGAS skipped: {scores.detail or 'unavailable'}_"
    return (
        "| Metric | Score |\n"
        "|--------|------:|\n"
        f"| Faithfulness | {_fmt(scores.faithfulness)} |\n"
        f"| Answer relevancy | {_fmt(scores.answer_relevancy)} |\n"
        f"| Context precision | {_fmt(scores.context_precision)} |\n"
        f"\nScored **{scores.n}** answers ({scores.detail})."
    )


def render_markdown(report: EvalReport) -> str:
    """Full before/after markdown report."""

    parts = [
        "# arXiv RAG — Evaluation Report",
        "",
        f"- Golden set: `{report.golden_path}`",
        f"- Corpus size: **{report.corpus_size}** chunks",
        "",
        "## Retrieval ablation",
        "",
        "Dense-only is the baseline. Hybrid adds BM25 via RRF. "
        "`hybrid_rerank` is the production stack (cross-encoder).",
        "",
        ablation_table(report.ablations),
        "",
    ]
    for row in report.ablations:
        parts.append(f"### `{row.name}`")
        parts.append("")
        parts.append(row.description)
        parts.append("")
        if row.scores.cases:
            parts.append("| Case | Hit | Recall | RR | Top IDs |")
            parts.append("|------|:---:|-------:|---:|---------|")
            for case in row.scores.cases:
                tops = ", ".join(f"`{i}`" for i in case.top_ids[:5]) or "—"
                parts.append(
                    f"| `{case.case_id}` | {'yes' if case.hit else 'no'} | "
                    f"{case.recall:.2f} | {case.reciprocal_rank:.2f} | {tops} |"
                )
            parts.append("")

    parts.extend(
        [
            "## Answer quality (RAGAS)",
            "",
            ragas_section(report.ragas),
            "",
            "## How to read this",
            "",
            "- **Hit-rate**: fraction of questions with >=1 relevant paper/chunk in top-k.",
            "- **Recall@k**: mean fraction of labeled relevant ids recovered.",
            "- **MRR**: mean reciprocal rank of the first relevant id.",
            "- **Purity@k**: mean fraction of top-k results that are labeled relevant "
            "(shows distractors even when hit-rate is already 1.0).",
            "- Expect `hybrid_rerank` >= `hybrid` >= `dense_only` on keyword-heavy questions.",
            "",
        ]
    )
    return "\n".join(parts)


def write_report(report: EvalReport, path: Path) -> None:
    """Write markdown + JSON sidecar next to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    sidecar = path.with_suffix(".json")
    sidecar.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"
