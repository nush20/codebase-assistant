"""HTTP client used by the Streamlit frontend to call the FastAPI backend."""

from typing import Any

import httpx

from config import API_BASE_URL, API_REQUEST_TIMEOUT_SECONDS, DEFAULT_TOP_K


class BackendError(RuntimeError):
    """A safe, user-facing error returned by or raised while calling the API."""


def _error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return detail if isinstance(detail, str) and detail else "The backend request failed."


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{API_BASE_URL}{path}", json=payload, timeout=API_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise BackendError(_error_message(exc.response)) from exc
    except (httpx.RequestError, ValueError) as exc:
        raise BackendError(
            f"Cannot reach the Codebase Assistant API at {API_BASE_URL}. "
            "Make sure FastAPI is running."
        ) from exc


def index_repository(repo_path: str) -> dict[str, Any]:
    return _post("/repositories/index", {"repo_path": repo_path})


def query_repository(
    repo_name: str, question: str, top_k: int = DEFAULT_TOP_K
) -> dict[str, Any]:
    return _post(
        "/query",
        {"repo_name": repo_name, "question": question, "top_k": top_k},
    )
