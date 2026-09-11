import subprocess
from pathlib import Path

import pytest

import repository_source


def test_local_repository_path_is_unchanged():
    with repository_source.repository_source(" /tmp/example ") as path:
        assert path == "/tmp/example"


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/owner/repo",
        "https://gitlab.com/owner/repo",
        "https://github.com/owner/repo/issues",
        "https://user@github.com/owner/repo",
    ],
)
def test_rejects_unsafe_or_unsupported_urls(url):
    with pytest.raises(ValueError):
        with repository_source.repository_source(url):
            pass


def test_clones_public_github_repository_shallowly(monkeypatch):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        Path(command[-1]).mkdir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(repository_source.subprocess, "run", fake_run)
    url = "https://github.com/pallets/flask.git"
    with repository_source.repository_source(url) as path:
        assert Path(path).name == "flask"
        assert Path(path).is_dir()

    assert recorded["command"][:6] == [
        "git", "clone", "--depth", "1", "--single-branch", "--",
    ]
    assert recorded["command"][6] == url
    assert recorded["kwargs"]["timeout"] == 120
