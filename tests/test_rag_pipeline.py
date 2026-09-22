from models import CodeChunk, RetrievedChunk
import rag_pipeline
from rag_pipeline import deduplicate_chunks, score_candidates, select_with_symbol_context


def result(path, symbol, text, score, index, language="Python"):
    chunk = CodeChunk(
        f"00000000-0000-0000-0000-0000000000{index:02d}", "repo", path, language,
        text, index, index + 2, symbol, "method", index,
    )
    return RetrievedChunk(chunk, score)


def test_exact_identifier_match_beats_higher_vector_similarity():
    candidates = [
        result("docs/commands.md", "docs/commands.md", "How commands work", 0.82, 1),
        result("typer/main.py", "Typer.command", "def command(self): pass", 0.60, 2),
    ]
    ranked = score_candidates("How does `Typer.command()` work?", candidates)
    assert ranked[0].chunk.unit_name == "Typer.command"


def test_score_is_vector_plus_identifier_and_token_overlap():
    candidate = result("main.py", "target", "def target(): pass", 0.5, 1)

    ranked = score_candidates("`target()`", [candidate])

    assert ranked[0].score == 1.0


def test_deduplication_keeps_distinct_windows_from_the_same_symbol():
    candidates = [
        result("main.py", "large", f"window {i}", 0.9 - i / 100, i)
        for i in range(1, 5)
    ] + [result("other.py", "helper", "helper", 0.5, 5)]
    ranked = deduplicate_chunks(score_candidates("How does large work?", candidates), 5)
    assert sum(item.chunk.unit_name == "large" for item in ranked) == 4
    assert any(item.chunk.unit_name == "helper" for item in ranked)


def test_deduplication_keeps_multiple_notebook_style_script_windows():
    candidates = [
        RetrievedChunk(
            CodeChunk(
                f"00000000-0000-0000-0000-0000000000{i:02d}",
                "repo", "analysis.py", "Python", f"step {i}", i * 10,
                i * 10 + 9, "analysis.py", "module_script", i,
            ),
            0.9 - i / 100,
        )
        for i in range(1, 4)
    ]

    ranked = deduplicate_chunks(
        score_candidates("How is the analysis performed?", candidates), 3
    )

    assert len(ranked) == 3


def test_deduplication_removes_repeated_text_even_with_different_ranges():
    candidates = [
        result("main.py", "first", "same implementation", 0.8, 1),
        result("main.py", "second", "same implementation", 0.7, 5),
    ]

    ranked = deduplicate_chunks(score_candidates("implementation", candidates), 2)

    assert len(ranked) == 1


def test_symbol_context_keeps_top_k_then_appends_adjacent_windows():
    candidates = [
        result("model.py", "GPT.generate", "middle", 0.9, 20),
        result("other.py", "helper", "helper", 0.8, 40),
        result("model.py", "GPT.generate", "first", 0.7, 10),
        result("model.py", "GPT.generate", "last", 0.6, 30),
    ]

    selected = select_with_symbol_context(candidates, 2)

    assert [item.chunk.text for item in selected] == [
        "middle", "helper", "first", "last",
    ]


def test_symbol_context_does_not_expand_module_windows():
    candidates = [
        RetrievedChunk(
            CodeChunk(
                f"00000000-0000-0000-0000-0000000000{i:02d}",
                "repo", "script.py", "Python", f"step {i}", i, i + 2,
                "script.py", "module_script", i,
            ),
            0.9 - i / 100,
        )
        for i in range(1, 4)
    ]

    selected = select_with_symbol_context(candidates, 2)

    assert [item.chunk.text for item in selected] == ["step 1", "step 2"]


def test_token_overlap_breaks_close_vector_similarity():
    candidates = [
        result("unrelated.py", "helper", "unrelated utility", 0.53, 1),
        result("routing.py", "register", "register CLI commands", 0.50, 2),
    ]
    ranked = score_candidates("How are CLI commands registered?", candidates)
    assert ranked[0].chunk.unit_name == "register"


def test_scoring_has_no_source_file_preference():
    candidates = [
        result("docs/server.md", "startServer", "server implementation", 0.60, 1, "Markdown"),
        result("src/server.mjs", "startServer", "server implementation", 0.60, 2, "JavaScript Module"),
    ]

    ranked = score_candidates("Where is the server implemented?", candidates)

    assert ranked[0].chunk.file_path == "docs/server.md"


def test_cloud_retrieval_sends_raw_question_without_local_embedding(monkeypatch):
    monkeypatch.setattr(rag_pipeline.vector_store, "cloud_inference", True)
    recorded = {}

    def search(query, repo_name, top_k):
        recorded.update(query=query, repo_name=repo_name, top_k=top_k)
        return []

    monkeypatch.setattr(rag_pipeline.vector_store, "search", search)
    monkeypatch.setattr(
        rag_pipeline,
        "embed_question",
        lambda _: (_ for _ in ()).throw(AssertionError("local embedding should not run")),
    )

    assert rag_pipeline.retrieve_chunks("repo", "Where is routing?", 6) == []
    assert recorded == {"query": "Where is routing?", "repo_name": "repo", "top_k": 100}


def test_retrieval_fetches_split_symbol_windows_outside_semantic_candidates(monkeypatch):
    seed = result("model.py", "GPT.generate", "def generate():", 0.8, 10)
    continuation = result(
        "model.py", "GPT.generate", "logits = logits / temperature", 0.0, 20
    )
    monkeypatch.setattr(rag_pipeline.vector_store, "cloud_inference", True)
    monkeypatch.setattr(
        rag_pipeline.vector_store, "search", lambda *_args, **_kwargs: [seed]
    )
    monkeypatch.setattr(
        rag_pipeline.vector_store,
        "get_symbol_chunks",
        lambda *_args, **_kwargs: [seed, continuation],
    )

    retrieved = rag_pipeline.retrieve_chunks(
        "repo", "How does GPT.generate() use temperature?", 1
    )

    assert [item.chunk.text for item in retrieved] == [
        "def generate():", "logits = logits / temperature",
    ]
