"""Public orchestration functions for repository ingestion and grounded Q&A."""

import logging
import re

from code_chunker import chunk_file
from config import (
    EXACT_IDENTIFIER_BOOST,
    FILENAME_TOKEN_COVERAGE_BOOST,
    IMPLEMENTATION_DOCS_PENALTY,
    IMPLEMENTATION_SOURCE_BOOST,
    IMPLEMENTATION_TEST_PENALTY,
    MAX_CHUNKS_PER_JAVA_FILE,
    MAX_CHUNKS_PER_SYMBOL,
    MIN_RETRIEVAL_CANDIDATES,
    NONIMPLEMENTATION_DOCS_PENALTY,
    NONIMPLEMENTATION_SOURCE_BOOST,
    NONIMPLEMENTATION_TEST_PENALTY,
    PARTIAL_IDENTIFIER_BOOST,
    PRIVATE_DIRECTORY_PENALTY,
    QUESTION_TOKEN_COVERAGE_BOOST,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    SHORT_SYMBOL_BOOST,
    TEXT_IDENTIFIER_BOOST,
    UNIT_TOKEN_COVERAGE_BOOST,
)
from embedder import embed_chunks, embed_question
from errors import (
    GenerationServiceError,
    IndexingServiceError,
    RepositoryNotIndexedError,
    RetrievalServiceError,
)
from llm_client import generate_answer
from models import AnswerResult, IngestionResult
from repo_loader import load_repository
from vector_store import vector_store

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CODE_IDENTIFIER_RE = re.compile(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
_STOP_WORDS = {"a", "an", "and", "are", "does", "how", "in", "is", "of", "the", "to", "what", "where", "which"}
_IRREGULAR_TOKEN_FORMS = {
    "built": "build",
    "made": "make",
    "sent": "send",
}
_IMPLEMENTATION_PHRASES = (
    "class", "code", "convert", "created", "executed", "generated", "handled",
    "implemented", "implementation", "integrate", "internally", "module", "parsed",
    "processed", "registered", "where", "which module",
)


def _word_forms(token: str) -> set[str]:
    """Return conservative forms for matching prose to code identifiers."""
    forms = {token}
    if token in _IRREGULAR_TOKEN_FORMS:
        forms.add(_IRREGULAR_TOKEN_FORMS[token])
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        forms.add(stem)
        forms.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        forms.add(stem)
        forms.add(stem + "e")
    return forms


def _tokens(value: str) -> set[str]:
    expanded = set()
    for raw_token in _TOKEN_RE.findall(value):
        pieces = [raw_token, *raw_token.split("_")]
        pieces.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw_token))
        for piece in pieces:
            token = piece.lower()
            if token and token not in _STOP_WORDS:
                for form in _word_forms(token):
                    expanded.add(form)
                    if len(form) > 3 and form.endswith("s") and not form.endswith("ss"):
                        expanded.add(form[:-1])
    return expanded


def _identifiers(question: str) -> set[str]:
    identifiers = set()
    for backticked, called in _CODE_IDENTIFIER_RE.findall(question):
        value = (backticked or called).strip().removesuffix("()")
        if value:
            identifiers.add(value.lower())
    identifiers.update(token.lower() for token in _TOKEN_RE.findall(question)
                       if "_" in token or (
                           not token.isupper() and any(char.isupper() for char in token[1:])
                       ))
    return identifiers


def rerank_chunks(question: str, candidates: list, top_k: int):
    """Rerank semantic candidates using code identifiers, paths, keywords, and diversity."""
    question_tokens = _tokens(question)
    identifiers = _identifiers(question)
    question_lower = question.lower()
    implementation_intent = any(phrase in question_lower for phrase in _IMPLEMENTATION_PHRASES)
    scored = []
    for item in candidates:
        chunk = item.chunk
        unit = chunk.unit_name.lower()
        short_unit = unit.rsplit(".", 1)[-1]
        path = chunk.file_path.lower()
        text_lower = chunk.text.lower()
        score = item.score

        for identifier in identifiers:
            identifier_short = identifier.rsplit(".", 1)[-1]
            if identifier in {unit, short_unit} or identifier_short in {unit, short_unit}:
                score += EXACT_IDENTIFIER_BOOST
            elif identifier in unit or identifier_short in unit:
                score += PARTIAL_IDENTIFIER_BOOST
            elif re.search(rf"\b{re.escape(identifier_short)}\b", text_lower):
                score += TEXT_IDENTIFIER_BOOST

        searchable_tokens = _tokens(f"{chunk.file_path} {chunk.unit_name} {chunk.text}")
        unit_tokens = _tokens(chunk.unit_name)
        short_symbol_tokens = _tokens(chunk.unit_name.rsplit(".", 1)[-1])
        filename_tokens = _tokens(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        if question_tokens:
            score += QUESTION_TOKEN_COVERAGE_BOOST * len(
                question_tokens & searchable_tokens
            ) / len(question_tokens)
            score += UNIT_TOKEN_COVERAGE_BOOST * len(
                question_tokens & unit_tokens
            ) / max(1, len(unit_tokens))
            score += FILENAME_TOKEN_COVERAGE_BOOST * len(
                question_tokens & filename_tokens
            ) / max(1, len(filename_tokens))
            if len(short_symbol_tokens) == 1 and short_symbol_tokens <= question_tokens:
                score += SHORT_SYMBOL_BOOST

        is_docs = path.startswith(("docs/", "docs_src/")) or "/docs/" in path
        is_test = path.startswith("tests/") or "/tests/" in path or path.startswith("test_")
        has_private_directory = any(
            part.startswith("_") for part in path.split("/")[:-1]
        )
        if implementation_intent:
            if path.endswith(".py") and not is_docs and not is_test:
                score += IMPLEMENTATION_SOURCE_BOOST
            if is_docs:
                score -= IMPLEMENTATION_DOCS_PENALTY
            if is_test:
                score -= IMPLEMENTATION_TEST_PENALTY
        else:
            if path.endswith(".py") and not is_docs and not is_test:
                score += NONIMPLEMENTATION_SOURCE_BOOST
            if is_docs:
                score -= NONIMPLEMENTATION_DOCS_PENALTY
            if is_test:
                score -= NONIMPLEMENTATION_TEST_PENALTY
        if implementation_intent and has_private_directory:
            score -= PRIVATE_DIRECTORY_PENALTY
        scored.append((score, item))

    selected = []
    symbol_counts: dict[tuple[str, str], int] = {}
    seen_ranges = set()
    for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True):
        chunk = item.chunk
        symbol_key = (chunk.file_path, chunk.unit_name)
        range_key = (chunk.file_path, chunk.start_line, chunk.end_line)
        symbol_limit = MAX_CHUNKS_PER_JAVA_FILE if path.endswith(".java") else MAX_CHUNKS_PER_SYMBOL
        if range_key in seen_ranges or symbol_counts.get(symbol_key, 0) >= symbol_limit:
            continue
        selected.append(item)
        seen_ranges.add(range_key)
        symbol_counts[symbol_key] = symbol_counts.get(symbol_key, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def ingest_repo(repo_path: str) -> IngestionResult:
    logger.info("Repository indexing started")
    loaded = load_repository(repo_path)
    chunks = []
    errors = list(loaded.errors)
    for file in loaded.files:
        try:
            chunks.extend(chunk_file(file))
        except Exception as exc:
            errors.append(f"{file.file_path}: chunking failed: {exc}")
    if not loaded.files:
        raise ValueError("No supported readable files were found in the repository.")
    if not chunks:
        raise ValueError("Supported files were found, but no non-empty code chunks could be created.")
    try:
        embeddings = None if vector_store.cloud_inference else embed_chunks(chunks)
        vector_store.clear_repository(loaded.repo_name)
        vector_store.upsert_chunks(chunks, embeddings)
    except Exception as exc:
        logger.exception("Repository indexing dependency failed for repository %s", loaded.repo_name)
        raise IndexingServiceError("Repository indexing failed.") from exc
    result = IngestionResult(
        loaded.repo_name, len(loaded.files), len(chunks), loaded.skipped_count, errors
    )
    logger.info(
        "Repository indexing completed for %s: files=%d chunks=%d skipped=%d",
        result.repo_name,
        result.files_processed,
        result.chunks_created,
        result.skipped_file_count,
    )
    return result


def retrieve_chunks(repo_name: str, question: str, top_k: int):
    candidate_k = max(MIN_RETRIEVAL_CANDIDATES, top_k * RETRIEVAL_CANDIDATE_MULTIPLIER)
    try:
        query = question if vector_store.cloud_inference else embed_question(question)
        candidates = vector_store.search(query, repo_name, candidate_k)
    except Exception as exc:
        logger.exception("Retrieval failed for repository %s", repo_name)
        raise RetrievalServiceError("Code retrieval failed.") from exc
    results = rerank_chunks(question, candidates, top_k)
    logger.info("Retrieval completed for %s: chunks=%d", repo_name, len(results))
    return results


def answer_question(repo_name: str, question: str, top_k: int = 6) -> AnswerResult:
    if not repo_name or not repo_name.strip():
        raise ValueError("Repository name cannot be empty.")
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    repo_name = repo_name.strip()
    try:
        indexed_chunks = vector_store.count_repository_chunks(repo_name)
    except Exception as exc:
        logger.exception("Unable to inspect index for repository %s", repo_name)
        raise RetrievalServiceError("Unable to access the repository index.") from exc
    if indexed_chunks == 0:
        raise RepositoryNotIndexedError(f"Repository is not indexed: {repo_name}")
    logger.info("Query received for repository %s", repo_name)
    retrieved = retrieve_chunks(repo_name, question.strip(), top_k)
    if not retrieved:
        return AnswerResult("No relevant indexed code chunks were found for this repository.", [])
    try:
        answer = generate_answer(question.strip(), retrieved)
    except Exception as exc:
        logger.exception("Answer generation failed for repository %s", repo_name)
        raise GenerationServiceError("Answer generation failed.") from exc
    logger.info("Answer generation completed for repository %s", repo_name)
    return AnswerResult(answer, retrieved)
