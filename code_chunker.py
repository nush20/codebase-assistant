"""Python AST-aware and generic line-window code chunking."""

from __future__ import annotations

import ast
import uuid

from config import (
    CHUNK_OVERLAP,
    CHUNK_WINDOW_SIZE,
    JAVA_CHUNK_OVERLAP,
    JAVA_CHUNK_WINDOW_SIZE,
)
from models import CodeChunk, RepositoryFile


def _chunk_id(file: RepositoryFile, start: int, end: int, index: int) -> str:
    value = f"{file.repo_name}:{file.file_path}:{start}:{end}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _windows(file: RepositoryFile, lines: list[str], base_line: int, unit_name: str,
             unit_type: str, window_size: int, overlap: int, start_index: int = 0) -> list[CodeChunk]:
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("window_size must be positive and overlap must be between 0 and window_size - 1")
    chunks: list[CodeChunk] = []
    step = window_size - overlap
    for offset in range(0, len(lines), step):
        window = lines[offset:offset + window_size]
        if not window:
            break
        text = "\n".join(window)
        if text.strip():
            start = base_line + offset
            end = start + len(window) - 1
            index = start_index + len(chunks)
            chunks.append(CodeChunk(
                _chunk_id(file, start, end, index), file.repo_name, file.file_path,
                file.language, text, start, end, unit_name, unit_type, index,
            ))
        if offset + window_size >= len(lines):
            break
    return chunks


def chunk_generic(file: RepositoryFile, window_size: int = CHUNK_WINDOW_SIZE,
                  overlap: int = CHUNK_OVERLAP) -> list[CodeChunk]:
    return _windows(file, file.content.splitlines(), 1, file.file_path, "line_window", window_size, overlap)


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _remaining_ranges(lines: list[str], start: int, end: int, excluded: set[int]) -> list[tuple[int, int]]:
    """Return non-empty source ranges after excluding child units."""
    ranges: list[tuple[int, int]] = []
    range_start = None
    last_nonblank = None
    for line_no in range(start, end + 1):
        if line_no in excluded:
            if range_start is not None and last_nonblank is not None:
                ranges.append((range_start, last_nonblank))
            range_start = last_nonblank = None
        elif lines[line_no - 1].strip():
            range_start = range_start or line_no
            last_nonblank = line_no
    if range_start is not None and last_nonblank is not None:
        ranges.append((range_start, last_nonblank))
    return ranges


def chunk_python(file: RepositoryFile, window_size: int = CHUNK_WINDOW_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> list[CodeChunk]:
    lines = file.content.splitlines()
    try:
        tree = ast.parse(file.content)
    except (SyntaxError, ValueError):
        return chunk_generic(file, window_size, overlap)

    units: list[tuple[int, int, str, str]] = []
    covered: set[int] = set()
    type_names = {ast.FunctionDef: "function", ast.AsyncFunctionDef: "async_function"}
    for node in tree.body:
        if type(node) in type_names:
            start = _node_start(node)
            end = node.end_lineno or node.lineno
            units.append((start, end, node.name, type_names[type(node)]))
            covered.update(range(start, end + 1))
        elif isinstance(node, ast.ClassDef):
            class_start = _node_start(node)
            class_end = node.end_lineno or node.lineno
            covered.update(range(class_start, class_end + 1))
            method_lines: set[int] = set()
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_start = _node_start(child)
                    method_end = child.end_lineno or child.lineno
                    method_type = "async_method" if isinstance(child, ast.AsyncFunctionDef) else "method"
                    units.append((
                        method_start, method_end, f"{node.name}.{child.name}",
                        method_type,
                    ))
                    method_lines.update(range(method_start, method_end + 1))
            for start, end in _remaining_ranges(lines, class_start, class_end, method_lines):
                units.append((start, end, node.name, "class"))

    module_ranges = _remaining_ranges(lines, 1, len(lines), covered)

    chunks: list[CodeChunk] = []
    ordered = units + [(start, end, file.file_path, "module") for start, end in module_ranges]
    for start, end, name, kind in sorted(ordered):
        unit_lines = lines[start - 1:end]
        chunks.extend(_windows(file, unit_lines, start, name, kind, window_size, overlap, len(chunks)))
    return chunks


def chunk_file(file: RepositoryFile, window_size: int = CHUNK_WINDOW_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[CodeChunk]:
    if file.extension == ".py":
        return chunk_python(file, window_size, overlap)
    if file.extension == ".java":
        return chunk_generic(file, JAVA_CHUNK_WINDOW_SIZE, JAVA_CHUNK_OVERLAP)
    return chunk_generic(file, window_size, overlap)
