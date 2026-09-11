"""Resolve local repository paths and safely clone public GitHub repositories."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


_GITHUB_PATH = re.compile(r"^/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$")


def _github_repository_name(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Only public HTTPS GitHub repository URLs are supported.")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise ValueError("The GitHub repository URL is invalid.")
    if not _GITHUB_PATH.fullmatch(parsed.path):
        raise ValueError("Use a GitHub URL in the form https://github.com/owner/repository.git")
    return Path(parsed.path.rstrip("/")).name.removesuffix(".git")


@contextmanager
def repository_source(value: str) -> Iterator[str]:
    """Yield a local directory for either a local path or public GitHub URL."""
    source = value.strip()
    if not source.startswith(("http://", "https://")):
        yield source
        return

    repository_name = _github_repository_name(source)
    with tempfile.TemporaryDirectory(prefix="codebase-assistant-") as temp_dir:
        destination = Path(temp_dir) / repository_name
        environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch", "--", source, str(destination)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise ValueError("Git is not installed on the server.") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Repository cloning timed out.") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError("The public GitHub repository could not be cloned.") from exc
        yield str(destination)
