"""Safe recursive loading of supported, textual repository files."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    IGNORED_DIRECTORIES,
    IGNORED_FILES,
    LANGUAGE_BY_EXTENSION,
    MAX_FILE_SIZE_BYTES,
    MAX_REPOSITORY_FILES,
    MAX_REPOSITORY_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)
from models import RepositoryFile


@dataclass
class LoadResult:
    repo_name: str
    files: list[RepositoryFile] = field(default_factory=list)
    skipped_count: int = 0
    errors: list[str] = field(default_factory=list)


def load_repository(
    repo_path: str,
    max_file_size: int = MAX_FILE_SIZE_BYTES,
    max_files: int = MAX_REPOSITORY_FILES,
    max_total_size: int = MAX_REPOSITORY_SIZE_BYTES,
) -> LoadResult:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    result = LoadResult(repo_name=root.name)
    total_size = 0
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith("."))
        for name in sorted(names):
            path = Path(current) / name
            extension = path.suffix.lower()
            if (
                name.startswith(".")
                or name in IGNORED_FILES
                or extension not in SUPPORTED_EXTENSIONS
            ):
                result.skipped_count += 1
                continue
            try:
                file_size = path.stat().st_size
                if file_size > max_file_size:
                    result.skipped_count += 1
                    continue
                if len(result.files) >= max_files:
                    raise ValueError(f"Repository exceeds the {max_files}-file indexing limit.")
                if total_size + file_size > max_total_size:
                    raise ValueError(
                        f"Repository exceeds the {max_total_size // 1_000_000} MB indexing limit."
                    )
                raw = path.read_bytes()
                if b"\x00" in raw:
                    result.skipped_count += 1
                    continue
                content = raw.decode("utf-8")
                result.files.append(RepositoryFile(
                    repo_name=root.name,
                    file_path=path.relative_to(root).as_posix(),
                    extension=extension,
                    language=LANGUAGE_BY_EXTENSION[extension],
                    content=content,
                ))
                total_size += file_size
            except (OSError, UnicodeDecodeError) as exc:
                result.skipped_count += 1
                result.errors.append(f"{path.relative_to(root)}: {exc}")
    return result
