"""Sentence-transformer embedding generation, loaded lazily once."""

from functools import lru_cache
from typing import Iterable

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME
from models import CodeChunk


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embedding_text(chunk: CodeChunk) -> str:
    symbol_parts = chunk.unit_name.replace(".", " ").replace("_", " ")
    return (f"File: {chunk.file_path}\nLanguage: {chunk.language}\nType: {chunk.unit_type}\n"
            f"Symbol: {chunk.unit_name}\nSymbol words: {symbol_parts}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n\nCode:\n{chunk.text}")


def embed_texts(texts: Iterable[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    items = list(texts)
    if not items:
        return []
    vectors = get_model().encode(items, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]:
    return embed_texts(embedding_text(chunk) for chunk in chunks)


def embed_question(question: str) -> list[float]:
    return embed_texts([question])[0]
