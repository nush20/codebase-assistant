from fastapi.testclient import TestClient

import api
from errors import GenerationServiceError, RepositoryNotIndexedError
from models import AnswerResult, CodeChunk, IngestionResult, RetrievedChunk


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_repository(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_repo",
        lambda path: IngestionResult("sample", 4, 12, 2, ["unreadable.py"]),
    )
    response = client.post("/repositories/index", json={"repo_path": "/tmp/sample"})
    assert response.status_code == 200
    assert response.json() == {
        "status": "indexed",
        "repo_name": "sample",
        "files_processed": 4,
        "chunks_created": 12,
        "skipped_file_count": 2,
        "indexing_errors": ["unreadable.py"],
    }


def test_query_returns_structured_sources(monkeypatch):
    chunk = CodeChunk(
        "id", "sample", "src/auth.py", "Python", "def login(): ...", 10, 12,
        "login", "function", 0,
    )
    monkeypatch.setattr(
        api,
        "answer_question",
        lambda repo, question, top_k: AnswerResult("Authentication is in login.", [RetrievedChunk(chunk, 0.91)]),
    )
    response = client.post(
        "/query", json={"repo_name": "sample", "question": "Where is authentication?"}
    )
    assert response.status_code == 200
    assert response.json()["sources"] == [{
        "file": "src/auth.py", "symbol": "login", "line_start": 10,
        "line_end": 12, "language": "Python", "unit_type": "function",
        "text": "def login(): ...", "score": 0.91,
    }]


def test_empty_query_is_rejected_before_pipeline():
    response = client.post("/query", json={"repo_name": "sample", "question": "   "})
    assert response.status_code == 422


def test_repository_not_indexed(monkeypatch):
    def fail(*_):
        raise RepositoryNotIndexedError("Repository is not indexed: missing")

    monkeypatch.setattr(api, "answer_question", fail)
    response = client.post("/query", json={"repo_name": "missing", "question": "Where?"})
    assert response.status_code == 409
    assert response.json() == {"detail": "Repository is not indexed: missing"}


def test_external_failure_is_sanitized(monkeypatch):
    def fail(*_):
        raise GenerationServiceError("secret provider detail")

    monkeypatch.setattr(api, "answer_question", fail)
    response = client.post("/query", json={"repo_name": "sample", "question": "Where?"})
    assert response.status_code == 502
    assert response.json() == {"detail": "A RAG dependency is unavailable."}


def test_unexpected_index_failure_is_sanitized(monkeypatch):
    def fail(_):
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(api, "ingest_repo", fail)
    response = client.post("/repositories/index", json={"repo_path": "/tmp/sample"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
