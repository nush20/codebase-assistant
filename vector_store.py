"""Qdrant storage with cloud configuration and a local in-memory fallback."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from config import (
    EMBEDDING_DIMENSION,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_INFERENCE_MODEL,
    QDRANT_URL,
)
from embedder import embedding_text
from models import CodeChunk, RetrievedChunk


class VectorStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
        cloud_inference: bool | None = None,
    ):
        self.cloud_inference = (
            bool(QDRANT_URL) if cloud_inference is None and client is None else bool(cloud_inference)
        )
        self.client = client if client is not None else self._default_client()
        self.collection_name = collection_name
        self._payload_index_ready = False

    @staticmethod
    def _default_client() -> QdrantClient:
        if QDRANT_URL:
            return QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
                cloud_inference=True,
                check_compatibility=False,
            )
        return QdrantClient(":memory:")

    def create_collection_if_needed(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
            )
        if self.cloud_inference and not self._payload_index_ready:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="repo_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            self._payload_index_ready = True

    def upsert_chunks(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        if not self.cloud_inference and embeddings is None:
            raise ValueError("Local embeddings are required when cloud inference is disabled")
        if embeddings is not None and len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match")
        self.create_collection_if_needed()
        vectors = (
            [
                models.Document(text=embedding_text(chunk), model=QDRANT_INFERENCE_MODEL)
                for chunk in chunks
            ]
            if self.cloud_inference
            else embeddings or []
        )
        points = [models.PointStruct(id=chunk.chunk_id, vector=vector, payload={
            "repo_name": chunk.repo_name, "file_path": chunk.file_path, "language": chunk.language,
            "text": chunk.text, "start_line": chunk.start_line, "end_line": chunk.end_line,
            "unit_name": chunk.unit_name, "unit_type": chunk.unit_type, "chunk_index": chunk.chunk_index,
        }) for chunk, vector in zip(chunks, vectors)]
        for start in range(0, len(points), 32):
            self.client.upsert(self.collection_name, points=points[start:start + 32], wait=True)

    def search(self, query: list[float] | str, repo_name: str, top_k: int) -> list[RetrievedChunk]:
        self.create_collection_if_needed()
        query_input = (
            models.Document(text=query, model=QDRANT_INFERENCE_MODEL)
            if self.cloud_inference and isinstance(query, str)
            else query
        )
        query_filter = models.Filter(must=[models.FieldCondition(
            key="repo_name", match=models.MatchValue(value=repo_name))])
        if hasattr(self.client, "query_points"):
            hits = self.client.query_points(self.collection_name, query=query_input,
                query_filter=query_filter, limit=top_k, with_payload=True).points
        else:  # Compatibility with older qdrant-client releases.
            hits = self.client.search(self.collection_name, query_vector=query_input,
                query_filter=query_filter, limit=top_k, with_payload=True)
        results = []
        for hit in hits:
            p = hit.payload or {}
            chunk = CodeChunk(str(hit.id), p["repo_name"], p["file_path"], p["language"], p["text"],
                p["start_line"], p["end_line"], p["unit_name"], p["unit_type"], p["chunk_index"])
            results.append(RetrievedChunk(chunk, float(hit.score)))
        return results

    def get_symbol_chunks(
        self, repo_name: str, file_path: str, unit_name: str
    ) -> list[RetrievedChunk]:
        """Load every stored window for one symbol without semantic filtering."""
        self.create_collection_if_needed()
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="repo_name", match=models.MatchValue(value=repo_name)
                ),
                models.FieldCondition(
                    key="file_path", match=models.MatchValue(value=file_path)
                ),
                models.FieldCondition(
                    key="unit_name", match=models.MatchValue(value=unit_name)
                ),
            ]),
            limit=64,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for record in records:
            payload = record.payload or {}
            chunk = CodeChunk(
                str(record.id), payload["repo_name"], payload["file_path"],
                payload["language"], payload["text"], payload["start_line"],
                payload["end_line"], payload["unit_name"], payload["unit_type"],
                payload["chunk_index"],
            )
            results.append(RetrievedChunk(chunk, 0.0))
        return sorted(results, key=lambda item: item.chunk.start_line)

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
