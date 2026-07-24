import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient

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
