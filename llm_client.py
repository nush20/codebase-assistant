"""Gemini-specific answer generation, isolated behind one function."""

import os
import time

from dotenv import load_dotenv

from config import (
    GEMINI_FALLBACK_MODEL_NAMES,
    GEMINI_MAX_ATTEMPTS_PER_MODEL,
    GEMINI_MODEL_NAME,
    GEMINI_RETRY_DELAY_SECONDS,
    MAX_RESPONSE_TOKENS,
)
from models import RetrievedChunk


def _build_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    excerpts = []
    for number, item in enumerate(retrieved_chunks, 1):
        chunk = item.chunk
        excerpts.append(
            f"SOURCE {number}\nFile: {chunk.file_path}\nSymbol: {chunk.unit_name}\n"
            f"Type: {chunk.unit_type}\nLines: {chunk.start_line}-{chunk.end_line}\n\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(excerpts)
    return f"""You are a codebase assistant. Answer only from the supplied code context.
Do not invent behavior that is absent from the excerpts. If the context is insufficient,
explicitly say so. Cite every code-based claim using exactly this format:
[path/to/file.py, symbol_name, lines 12-47]

USER QUESTION
{question}

RETRIEVED CODE CONTEXT
{context}
"""


def _is_transient_error(exc: Exception) -> bool:
    """Return whether retrying a Gemini request is likely to help."""
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    message = str(exc).upper()
    return code in (429, 503) or status_code in (429, 503) or any(
        marker in message for marker in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")
    )


def generate_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your environment or .env file.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(question, retrieved_chunks)
    generation_config = types.GenerateContentConfig(
        max_output_tokens=MAX_RESPONSE_TOKENS, temperature=0.1
    )
    model_names = (GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODEL_NAMES)
    last_error: Exception | None = None
    for model_name in model_names:
        for attempt in range(GEMINI_MAX_ATTEMPTS_PER_MODEL):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=generation_config
                )
                if not response.text:
                    raise RuntimeError("Gemini returned no answer text.")
                return response.text
            except Exception as exc:
                if not _is_transient_error(exc):
                    raise
                last_error = exc
                if attempt + 1 < GEMINI_MAX_ATTEMPTS_PER_MODEL:
                    time.sleep(GEMINI_RETRY_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError(
        "Gemini is temporarily unavailable after retrying the configured models. "
        "Please try again shortly."
    ) from last_error
