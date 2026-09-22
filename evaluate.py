"""Command-line evaluation of retrieval and optional Gemini answer generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from llm_client import generate_answer
from rag_pipeline import ingest_repo, retrieve_chunks


def chunk_matches_expected(chunk: Any, expected: dict[str, Any]) -> bool:
    """Match a chunk against an expected file, symbol, and optional line range."""
    if chunk.file_path != expected["file_path"]:
        return False
    symbol = expected.get("symbol")
    if symbol and not (
        chunk.unit_name == symbol or chunk.unit_name.startswith(f"{symbol}.")
    ):
        return False
    expected_start = expected.get("line_start")
    expected_end = expected.get("line_end")
    if expected_start is not None and chunk.end_line < expected_start:
        return False
    if expected_end is not None and chunk.start_line > expected_end:
        return False
    return True


def score_retrieval(
    retrieved: list[Any],
    expected_sources: list[dict[str, Any]],
    accepted_sources: list[dict[str, Any]] | None = None,
) -> dict[str, float | int]:
    """Score primary-source recall while accepting other valid supporting evidence."""
    accepted_sources = accepted_sources or []
    relevant_sources = [*expected_sources, *accepted_sources]
    matched_expected = {
        index
        for index, expected in enumerate(expected_sources)
        if any(chunk_matches_expected(item.chunk, expected) for item in retrieved)
    }
    relevant_ranks = [
        rank
        for rank, item in enumerate(retrieved, 1)
        if any(chunk_matches_expected(item.chunk, expected) for expected in relevant_sources)
    ]
    relevant_chunks = len(relevant_ranks)
    return {
        "hit": int(bool(relevant_ranks)),
        "primary_hit": int(bool(matched_expected)),
        "recall": len(matched_expected) / len(expected_sources) if expected_sources else 1.0,
        "precision": relevant_chunks / len(retrieved) if retrieved else 0.0,
        "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "matched_expected_sources": len(matched_expected),
        "expected_source_count": len(expected_sources),
    }


def evaluate(
    repo_path: str,
    dataset_path: str,
    top_k: int,
    generate: bool = False,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    questions = dataset.get("questions", [])
    if not questions:
        raise ValueError("Evaluation dataset must contain a non-empty 'questions' list.")

    cutoffs = sorted(set(k_values or [1, 3, top_k]) | {top_k})
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("All evaluation k values must be at least 1.")
    max_k = max(cutoffs)

    ingestion = ingest_repo(repo_path)
    results = []
    for case in questions:
        retrieved = retrieve_chunks(ingestion.repo_name, case["question"], max_k)
        expected_sources = case.get("expected_sources", [])
        accepted_sources = case.get("accepted_sources", [])
        scores_by_k = {
            str(k): score_retrieval(retrieved[:k], expected_sources, accepted_sources)
            for k in cutoffs
        }
        scores = scores_by_k[str(top_k)]
        result = {
            "id": case["id"],
            "question": case["question"],
            "category": case.get("category", "unspecified"),
            "expected_sources": case.get("expected_sources", []),
            "accepted_sources": accepted_sources,
            "important_facts": case.get("important_facts", []),
            "scores": scores,
            "scores_by_k": scores_by_k,
            "retrieved": [
                {
                    "rank": rank,
                    "score": item.score,
                    "file_path": item.chunk.file_path,
                    "symbol": item.chunk.unit_name,
                    "lines": [item.chunk.start_line, item.chunk.end_line],
                }
                for rank, item in enumerate(retrieved, 1)
            ],
        }
        if generate:
            result["answer"] = generate_answer(case["question"], retrieved)
            result["manual_answer_score"] = {
                "correctness_0_to_2": None,
                "groundedness_0_to_2": None,
                "completeness_0_to_2": None,
                "citations_0_to_2": None,
                "relevance_0_to_2": None,
                "uncertainty_0_to_2": None,
            }
        results.append(result)

    metrics_by_k = {}
    for k in cutoffs:
        key = str(k)
        metrics_by_k[key] = {
            "hit_rate": mean(item["scores_by_k"][key]["hit"] for item in results),
            "primary_hit_rate": mean(
                item["scores_by_k"][key]["primary_hit"] for item in results
            ),
            "mean_recall": mean(item["scores_by_k"][key]["recall"] for item in results),
            "mean_precision": mean(item["scores_by_k"][key]["precision"] for item in results),
            "mean_reciprocal_rank": mean(
                item["scores_by_k"][key]["reciprocal_rank"] for item in results
            ),
        }
    top_metrics = metrics_by_k[str(top_k)]
    summary = {
        "question_count": len(results),
        "top_k": top_k,
        "evaluated_k_values": cutoffs,
        "metrics_by_k": metrics_by_k,
        "hit_rate": top_metrics["hit_rate"],
        "mean_recall": top_metrics["mean_recall"],
        "mean_precision": top_metrics["mean_precision"],
        "mean_reciprocal_rank": top_metrics["mean_reciprocal_rank"],
    }
    return {"repository": asdict(ingestion), "summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate codebase-assistant retrieval and answers.")
    parser.add_argument("repo_path", help="Local repository directory")
    parser.add_argument("dataset", help="JSON evaluation dataset")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--k-values", default="1,3,6",
        help="Comma-separated retrieval cutoffs to report, for example 1,3,6",
    )
    parser.add_argument("--generate", action="store_true", help="Also call Gemini and include answers")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    except ValueError:
        parser.error("--k-values must be comma-separated positive integers")
    if not k_values or any(value < 1 for value in k_values):
        parser.error("--k-values must contain positive integers")

    report = evaluate(args.repo_path, args.dataset, args.top_k, args.generate, k_values)
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(f"Questions: {summary['question_count']}")
    print("k  UsefulHit PrimaryHit Recall    Precision MRR")
    for k in summary["evaluated_k_values"]:
        metrics = summary["metrics_by_k"][str(k)]
        print(
            f"{k:<2} {metrics['hit_rate']:>9.1%} {metrics['primary_hit_rate']:>10.1%} "
            f"{metrics['mean_recall']:>9.1%} "
            f"{metrics['mean_precision']:>9.1%} {metrics['mean_reciprocal_rank']:>5.3f}"
        )
    print(f"Detailed report: {output.resolve()}")


if __name__ == "__main__":
    main()
