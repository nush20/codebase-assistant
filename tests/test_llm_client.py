from types import SimpleNamespace

import llm_client


class BusyError(Exception):
    status_code = 503


def test_transient_error_detection():
    assert llm_client._is_transient_error(BusyError("high demand"))
    assert llm_client._is_transient_error(Exception("429 RESOURCE_EXHAUSTED"))
    assert not llm_client._is_transient_error(Exception("400 invalid request"))


def test_retries_then_uses_fallback(monkeypatch):
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(model)
            if model == "primary":
                raise BusyError("temporarily unavailable")
            return SimpleNamespace(text="fallback answer")

    monkeypatch.setattr(llm_client, "GEMINI_MODEL_NAME", "primary")
    monkeypatch.setattr(llm_client, "GEMINI_FALLBACK_MODEL_NAMES", ("fallback",))
    monkeypatch.setattr(llm_client, "GEMINI_MAX_ATTEMPTS_PER_MODEL", 2)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: SimpleNamespace(models=Models()))

    assert llm_client.generate_answer("question", []) == "fallback answer"
    assert calls == ["primary", "primary", "fallback"]
