import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient
from qdrant_client import models

from models import CodeChunk
from vector_store import VectorStore


def chunk(repo, path, index):
    return CodeChunk(f"00000000-0000-0000-0000-00000000000{index}", repo, path, "Python", path, 1, 1, "unit", "function", index)


def vector(position):
    result = [0.0] * 384
    result[position] = 1.0
    return result


def test_insert_search_filter_count_and_clear():
    store = VectorStore(QdrantClient(":memory:"), "test_chunks")
    a, b = chunk("repo-a", "a.py", 1), chunk("repo-b", "b.py", 2)
    store.upsert_chunks([a, b], [vector(0), vector(1)])
    assert store.count_repository_chunks("repo-a") == 1
    assert store.search(vector(0), "repo-a", 5)[0].chunk.file_path == "a.py"
    assert store.search(vector(0), "repo-b", 5)[0].chunk.file_path == "b.py"
    store.clear_repository("repo-a")
    assert store.count_repository_chunks("repo-a") == 0
    assert store.count_repository_chunks("repo-b") == 1


class CloudInferenceClient:
    def __init__(self):
        self.upserted = []
        self.query = None
        self.payload_indexes = []

    def collection_exists(self, _):
        return True

    def upsert(self, _, points, wait):
        self.upserted.extend(points)

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)

    def query_points(self, _, query, **__):
        self.query = query
        return type("Response", (), {"points": []})()


def test_cloud_inference_sends_chunk_text_and_query_to_qdrant():
    client = CloudInferenceClient()
    store = VectorStore(client, "cloud_chunks", cloud_inference=True)
    item = chunk("repo", "src/example.py", 1)

    store.upsert_chunks([item])
    store.search("Where is the example?", "repo", 5)

    assert isinstance(client.upserted[0].vector, models.Document)
    assert "src/example.py" in client.upserted[0].vector.text
    assert client.upserted[0].vector.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert isinstance(client.query, models.Document)
    assert client.query.text == "Where is the example?"
    assert client.payload_indexes == [{
        "collection_name": "cloud_chunks",
        "field_name": "repo_name",
        "field_schema": models.PayloadSchemaType.KEYWORD,
        "wait": True,
    }]


def test_local_store_requires_explicit_embeddings():
    store = VectorStore(QdrantClient(":memory:"), "local_chunks")
    with pytest.raises(ValueError, match="Local embeddings"):
        store.upsert_chunks([chunk("repo", "a.py", 1)])
