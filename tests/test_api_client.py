import httpx
import pytest

import api_client


def test_index_repository_calls_backend(monkeypatch):
    request = httpx.Request("POST", "http://test/repositories/index")
    response = httpx.Response(200, request=request, json={"repo_name": "flask"})
    monkeypatch.setattr(api_client.httpx, "post", lambda *args, **kwargs: response)

    assert api_client.index_repository("/repos/flask") == {"repo_name": "flask"}


def test_query_repository_sends_repository_question_and_top_k(monkeypatch):
    captured = {}

    def post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"answer": "ok"})

    monkeypatch.setattr(api_client.httpx, "post", post)
    assert api_client.query_repository("flask", "How?", 5) == {"answer": "ok"}
    assert captured["json"] == {"repo_name": "flask", "question": "How?", "top_k": 5}


def test_backend_error_uses_safe_api_detail(monkeypatch):
    request = httpx.Request("POST", "http://test/query")
    response = httpx.Response(409, request=request, json={"detail": "Repository is not indexed"})
    monkeypatch.setattr(api_client.httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(api_client.BackendError, match="Repository is not indexed"):
        api_client.query_repository("missing", "How?")


def test_connection_error_explains_how_to_recover(monkeypatch):
    def fail(url, **kwargs):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(api_client.httpx, "post", fail)
    with pytest.raises(api_client.BackendError, match="Make sure FastAPI is running"):
        api_client.index_repository("/repos/flask")
