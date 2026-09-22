import uuid

import pytest

from code_chunker import chunk_file, chunk_generic, chunk_python
from models import RepositoryFile


def py_file(text):
    return RepositoryFile("repo", "sample.py", ".py", "Python", text)


def test_python_units_module_content_and_ranges():
    text = "import os\nVALUE = 1\n\ndef hello():\n    return VALUE\n\nasync def later():\n    return 2\n\nclass Box:\n    def get(self):\n        return 3\n"
    chunks = chunk_python(py_file(text), window_size=40, overlap=5)
    by_type = {chunk.unit_type: chunk for chunk in chunks}
    assert by_type["function"].unit_name == "hello" and (by_type["function"].start_line, by_type["function"].end_line) == (4, 5)
    assert by_type["async_function"].unit_name == "later"
    assert by_type["class"].unit_name == "Box" and "class Box" in by_type["class"].text
    assert by_type["method"].unit_name == "Box.get" and "def get" in by_type["method"].text
    assert "def get" not in by_type["class"].text
    assert "import os" in by_type["module"].text and "VALUE = 1" in by_type["module"].text
    assert all(str(uuid.UUID(chunk.chunk_id)) == chunk.chunk_id for chunk in chunks)


def test_syntax_error_falls_back_to_windows():
    chunks = chunk_python(py_file("def broken(:\n  pass\n"), window_size=2, overlap=0)
    assert chunks[0].unit_type == "line_window"


def test_large_function_splits_with_correct_ranges():
    text = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(10))
    chunks = chunk_python(py_file(text), window_size=5, overlap=1)
    assert len(chunks) == 3
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 5), (5, 9), (9, 11)]
    assert all(c.unit_name == "big" for c in chunks)


def test_decorated_sync_and_async_methods_have_qualified_names_without_duplication():
    text = "class Service:\n    label = 'x'\n\n    @registered\n    def run(self):\n        return 1\n\n    async def wait(self):\n        return 2\n"
    chunks = chunk_python(py_file(text), window_size=40, overlap=5)
    methods = {chunk.unit_name: chunk for chunk in chunks if chunk.unit_type in {"method", "async_method"}}
    assert (methods["Service.run"].start_line, methods["Service.run"].end_line) == (4, 6)
    assert methods["Service.run"].text.startswith("    @registered")
    assert methods["Service.wait"].unit_type == "async_method"
    combined_class_text = "\n".join(chunk.text for chunk in chunks if chunk.unit_type == "class")
    assert "label = 'x'" in combined_class_text
    assert "def run" not in combined_class_text and "def wait" not in combined_class_text




def test_generic_overlap_ranges_and_no_blank_chunks():
    file = RepositoryFile("repo", "notes.txt", ".txt", "Text", "1\n2\n3\n4\n5\n6\n")
    chunks = chunk_generic(file, window_size=4, overlap=1)
    assert [c.text for c in chunks] == ["1\n2\n3\n4", "4\n5\n6"]
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 6)]
    blank = RepositoryFile("repo", "blank.txt", ".txt", "Text", "  \n\n")
    assert chunk_generic(blank, 2, 0) == []


@pytest.mark.parametrize(
    ("path", "extension", "language", "content", "expected_symbol"),
    [
        ("main.js", ".js", "JavaScript", "function run() { return 1; }", "run"),
        ("index.mjs", ".mjs", "JavaScript Module", "export function run() { return 1; }", "run"),
        ("Main.java", ".java", "Java", "class Box { int get() { return 1; } }", "Box.get"),
        ("main.go", ".go", "Go", "package main\nfunc run() int { return 1 }", "run"),
        ("main.c", ".c", "C", "int run(void) { return 1; }", "run"),
        ("main.rs", ".rs", "Rust", "fn run() -> i32 { 1 }", "run"),
    ],
)
def test_tree_sitter_extracts_symbols_across_languages(
    path, extension, language, content, expected_symbol
):
    file = RepositoryFile("repo", path, extension, language, content)

    chunks = chunk_file(file)

    assert any(chunk.unit_name == expected_symbol for chunk in chunks)


def test_non_code_file_uses_generic_line_windows():
    file = RepositoryFile("repo", "README.md", ".md", "Markdown", "one\ntwo\n")

    chunks = chunk_file(file)

    assert len(chunks) == 1
    assert chunks[0].unit_type == "line_window"


@pytest.mark.parametrize(
    ("path", "extension", "language"),
    [
        ("index.html", ".html", "HTML"),
        ("styles.css", ".css", "CSS"),
        ("theme.scss", ".scss", "SCSS"),
        ("theme.sass", ".sass", "Sass"),
    ],
)
def test_web_markup_and_styles_use_generic_chunks(path, extension, language):
    file = RepositoryFile("repo", path, extension, language, "one\ntwo\n")

    chunks = chunk_file(file)

    assert len(chunks) == 1
    assert chunks[0].unit_type == "line_window"


def test_javascript_call_callback_gets_a_useful_symbol_name():
    file = RepositoryFile(
        "repo",
        "api/index.mjs",
        ".mjs",
        "JavaScript Module",
        'app.get("/users", async (req, res) => {\n  return res.json([]);\n});\n',
    )

    chunks = chunk_file(file)

    callback = next(chunk for chunk in chunks if chunk.unit_name == 'app.get"/users"')
    assert callback.unit_type == "async_function"
    assert (callback.start_line, callback.end_line) == (1, 3)
