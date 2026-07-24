from models import CodeChunk, RetrievedChunk
from rag_pipeline import _tokens, rerank_chunks


def result(path, symbol, text, score, index):
    chunk = CodeChunk(
        f"00000000-0000-0000-0000-0000000000{index:02d}", "repo", path, "Python",
        text, index, index + 2, symbol, "method", index,
    )
    return RetrievedChunk(chunk, score)


def test_exact_symbol_and_implementation_path_beat_semantic_docs_result():
    candidates = [
        result("docs/commands.md", "docs/commands.md", "How commands work", 0.82, 1),
        result("typer/main.py", "Typer.command", "def command(self): pass", 0.60, 2),
    ]
    ranked = rerank_chunks("How does `app.command()` work internally?", candidates, 2)
    assert ranked[0].chunk.unit_name == "Typer.command"


def test_reranker_limits_duplicate_symbol_windows():
    candidates = [
        result("main.py", "large", f"window {i}", 0.9 - i / 100, i)
        for i in range(1, 5)
    ] + [result("other.py", "helper", "helper", 0.5, 5)]
    ranked = rerank_chunks("How does large work?", candidates, 5)
    assert sum(item.chunk.unit_name == "large" for item in ranked) == 1
    assert any(item.chunk.unit_name == "helper" for item in ranked)


def test_token_forms_match_natural_language_to_code_verbs():
    tokens = _tokens("requests were sent while parsing and building")
    assert {"send", "parse", "build"} <= tokens


def test_exact_short_symbol_beats_higher_semantic_cli_result():
    candidates = [
        result("typer/testing.py", "CliRunner", "CLI commands", 0.53, 1),
        result("typer/main.py", "Typer.command", "def command(self): pass", 0.39, 2),
    ]
    ranked = rerank_chunks("How are CLI commands registered?", candidates, 2)
    assert ranked[0].chunk.unit_name == "Typer.command"
