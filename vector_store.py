"""Qdrant storage with cloud configuration and a local in-memory fallback."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from config import EMBEDDING_DIMENSION, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL
from models import CodeChunk, RetrievedChunk


class VectorStore:
    def __init__(self, client: QdrantClient | None = None, collection_name: str = QDRANT_COLLECTION_NAME):
        self.client = client if client is not None else self._default_client()
        self.collection_name = collection_name

    @staticmethod
    def _default_client() -> QdrantClient:
        if QDRANT_URL:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        return QdrantClient(":memory:")

    def create_collection_if_needed(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match")
        self.create_collection_if_needed()
        points = [models.PointStruct(id=chunk.chunk_id, vector=vector, payload={
            "repo_name": chunk.repo_name, "file_path": chunk.file_path, "language": chunk.language,
            "text": chunk.text, "start_line": chunk.start_line, "end_line": chunk.end_line,
            "unit_name": chunk.unit_name, "unit_type": chunk.unit_type, "chunk_index": chunk.chunk_index,
        }) for chunk, vector in zip(chunks, embeddings)]
        if points:
            self.client.upsert(self.collection_name, points=points, wait=True)

    def search(self, query_embedding: list[float], repo_name: str, top_k: int) -> list[RetrievedChunk]:
        self.create_collection_if_needed()
        query_filter = models.Filter(must=[models.FieldCondition(
            key="repo_name", match=models.MatchValue(value=repo_name))])
        if hasattr(self.client, "query_points"):
            hits = self.client.query_points(self.collection_name, query=query_embedding,
                query_filter=query_filter, limit=top_k, with_payload=True).points
        else:  # Compatibility with older qdrant-client releases.
            hits = self.client.search(self.collection_name, query_vector=query_embedding,
                query_filter=query_filter, limit=top_k, with_payload=True)
        results = []
        for hit in hits:
            p = hit.payload or {}
            chunk = CodeChunk(str(hit.id), p["repo_name"], p["file_path"], p["language"], p["text"],
                p["start_line"], p["end_line"], p["unit_name"], p["unit_type"], p["chunk_index"])
            results.append(RetrievedChunk(chunk, float(hit.score)))
        return results

    def clear_repository(self, repo_name: str) -> None:
        self.create_collection_if_needed()
        self.client.delete(self.collection_name, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="repo_name", match=models.MatchValue(value=repo_name))])), wait=True)

    def count_repository_chunks(self, repo_name: str) -> int:
        self.create_collection_if_needed()
        result = self.client.count(self.collection_name, count_filter=models.Filter(must=[
            models.FieldCondition(key="repo_name", match=models.MatchValue(value=repo_name))]), exact=True)
        return result.count


vector_store = VectorStore()
