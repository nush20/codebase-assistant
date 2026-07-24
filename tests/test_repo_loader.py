from repo_loader import load_repository


def test_loads_valid_relative_paths_and_skips_irrelevant_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not supported")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("bad()", encoding="utf-8")
    result = load_repository(str(tmp_path))
    assert [file.file_path for file in result.files] == ["src/main.py"]
    assert result.files[0].content == "print('ok')\n"
    assert result.skipped_count == 1


def test_rejects_invalid_path(tmp_path):
    try:
        load_repository(str(tmp_path / "missing"))
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
