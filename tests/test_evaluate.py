from types import SimpleNamespace

import json

import evaluate as evaluation_module
from evaluate import chunk_matches_expected, evaluate, score_retrieval
from models import IngestionResult


def item(path, symbol):
    return SimpleNamespace(
        score=0.5,
        chunk=SimpleNamespace(
            file_path=path, unit_name=symbol, start_line=1, end_line=2,
        ),
    )


def test_score_retrieval_metrics():
    retrieved = [item("wrong.py", "other"), item("typer/main.py", "Typer"), item("typer/main.py", "Typer")]
    expected = [
        {"file_path": "typer/main.py", "symbol": "Typer"},
        {"file_path": "typer/utils.py", "symbol": "get_params_from_function"},
    ]
    scores = score_retrieval(retrieved, expected)
    assert scores["hit"] == 1
    assert scores["recall"] == 0.5
    assert scores["precision"] == 2 / 3
    assert scores["reciprocal_rank"] == 0.5


def test_file_only_expectation_matches_any_symbol():
    scores = score_retrieval([item("typer/main.py", "get_command")], [{"file_path": "typer/main.py"}])
    assert scores["hit"] == 1
    assert scores["recall"] == 1.0


def test_class_expectation_matches_qualified_method_chunk():
    scores = score_retrieval(
        [item("typer/main.py", "Typer.__init__")],
        [{"file_path": "typer/main.py", "symbol": "Typer"}],
    )
    assert scores["hit"] == 1


def test_expected_line_range_must_overlap_retrieved_chunk():
    chunk = item("main.py", "target").chunk

    assert chunk_matches_expected(
        chunk, {"file_path": "main.py", "symbol": "target", "line_start": 2, "line_end": 4}
    )
    assert not chunk_matches_expected(
        chunk, {"file_path": "main.py", "symbol": "target", "line_start": 10, "line_end": 12}
    )


def test_accepted_alternative_counts_as_hit_but_not_primary_recall():
    scores = score_retrieval(
        [item("typer/rich_utils.py", "rich_format_help")],
        [{"file_path": "typer/core.py", "symbol": "TyperCommand"}],
        [{"file_path": "typer/rich_utils.py", "symbol": "rich_format_help"}],
    )
    assert scores["hit"] == 1
    assert scores["primary_hit"] == 0
    assert scores["recall"] == 0.0
    assert scores["precision"] == 1.0


def test_evaluate_reports_metrics_at_multiple_k(tmp_path, monkeypatch):
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps({"questions": [{
        "id": "q1", "question": "Where?",
        "expected_sources": [{"file_path": "correct.py", "symbol": "target"}],
    }]}))
    fake_ingestion = IngestionResult("repo", 1, 3, 0, [])
    retrieved = [item("wrong.py", "other"), item("correct.py", "target"), item("extra.py", "extra")]
    monkeypatch.setattr(evaluation_module, "ingest_repo", lambda _: fake_ingestion)
    monkeypatch.setattr(evaluation_module, "retrieve_chunks", lambda repo, question, k: retrieved[:k])

    report = evaluate("repo", str(dataset), top_k=3, k_values=[1, 3])
    assert report["summary"]["metrics_by_k"]["1"]["hit_rate"] == 0.0
    assert report["summary"]["metrics_by_k"]["3"]["hit_rate"] == 1.0
    assert report["results"][0]["scores_by_k"]["3"]["recall"] == 1.0
