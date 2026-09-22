"""Public orchestration functions for repository ingestion and grounded Q&A."""

import logging
import re

from code_chunker import chunk_file
from config import (
    IDENTIFIER_MATCH_WEIGHT,
    MIN_RETRIEVAL_CANDIDATES,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    TOKEN_OVERLAP_WEIGHT,
)
from embedder import embed_chunks, embed_question
from errors import (
    GenerationServiceError,
    IndexingServiceError,
    RepositoryNotIndexedError,
    RetrievalServiceError,
)
from llm_client import generate_answer
from models import AnswerResult, IngestionResult, RetrievedChunk
from repo_loader import load_repository
from vector_store import vector_store

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CODE_IDENTIFIER_RE = re.compile(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
_STOP_WORDS = {"a", "an", "and", "are", "does", "how", "in", "is", "of", "the", "to", "what", "where", "which"}
_CALLABLE_UNIT_TYPES = {"function", "async_function", "method", "async_method"}
_MAX_CONTEXT_CHUNKS_PER_SYMBOL = 3
_MAX_ADDITIONAL_CONTEXT_CHUNKS = 4


def _tokens(value: str) -> set[str]:
    tokens = set()
    for raw_token in _TOKEN_RE.findall(value):
        pieces = [raw_token, *raw_token.split("_")]
        pieces.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw_token))
        for piece in pieces:
            token = piece.lower()
            if token and token not in _STOP_WORDS:
                tokens.add(token)
    return tokens


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


def score_candidates(question: str, candidates: list):
    """Add only exact-identifier and token-overlap signals to vector similarity."""
    question_tokens = _tokens(question)
    identifiers = _identifiers(question)
    scored = []
    for item in candidates:
        chunk = item.chunk
        unit = chunk.unit_name.lower()
        short_unit = unit.rsplit(".", 1)[-1]
        identifier_match = any(
            identifier in {unit, short_unit}
            or identifier.rsplit(".", 1)[-1] in {unit, short_unit}
            for identifier in identifiers
        )
        searchable_tokens = _tokens(f"{chunk.file_path} {chunk.unit_name} {chunk.text}")
        token_overlap = (
            len(question_tokens & searchable_tokens) / len(question_tokens)
            if question_tokens else 0.0
        )
        score = (
            item.score
            + IDENTIFIER_MATCH_WEIGHT * float(identifier_match)
            + TOKEN_OVERLAP_WEIGHT * token_overlap
        )
        scored.append(RetrievedChunk(item.chunk, score))
    return sorted(scored, key=lambda item: item.score, reverse=True)


def deduplicate_chunks(scored_candidates: list, top_k: int):
    """Remove exact duplicate ranges or text after ranking."""
    selected = []
    seen_ranges = set()
    seen_text = set()
    for item in scored_candidates:
        chunk = item.chunk
        range_key = (chunk.file_path, chunk.start_line, chunk.end_line)
        text_key = " ".join(chunk.text.split())
        if range_key in seen_ranges or text_key in seen_text:
            continue
        selected.append(item)
        seen_ranges.add(range_key)
        seen_text.add(text_key)
        if len(selected) == top_k:
            break
    return selected


def select_with_symbol_context(ranked_candidates: list, top_k: int):
    """Keep the top-k ranking intact, then append nearby callable windows."""
    if top_k < 1:
        return []
    unique_candidates = deduplicate_chunks(ranked_candidates, len(ranked_candidates))
    by_symbol: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for item in unique_candidates:
        chunk = item.chunk
        if chunk.unit_type in _CALLABLE_UNIT_TYPES:
            by_symbol.setdefault((chunk.file_path, chunk.unit_name), []).append(item)
    for chunks in by_symbol.values():
        chunks.sort(key=lambda item: item.chunk.start_line)

    selected = unique_candidates[:top_k]
    selected_ids = {item.chunk.chunk_id for item in selected}
    additional = 0
    for seed in selected[:top_k]:
        symbol_key = (seed.chunk.file_path, seed.chunk.unit_name)
        symbol_chunks = by_symbol.get(symbol_key, [])
        if len(symbol_chunks) > 1:
            neighbours = sorted(
                (item for item in symbol_chunks if item.chunk.chunk_id != seed.chunk.chunk_id),
                key=lambda item: abs(item.chunk.start_line - seed.chunk.start_line),
            )[: _MAX_CONTEXT_CHUNKS_PER_SYMBOL - 1]
        else:
            neighbours = []
        for item in neighbours:
            if item.chunk.chunk_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.chunk.chunk_id)
            additional += 1
            if additional == _MAX_ADDITIONAL_CONTEXT_CHUNKS:
                return selected
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
    scored_candidates = score_candidates(question, candidates)
    results = select_with_symbol_context(scored_candidates, top_k)
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
