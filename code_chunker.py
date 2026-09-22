"""Tree-sitter-aware source chunking with a generic line-window fallback."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import uuid

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from config import (
    CHUNK_OVERLAP,
    CHUNK_WINDOW_SIZE,
    TREE_SITTER_UNIT_OVERLAP,
    TREE_SITTER_UNIT_WINDOW_SIZE,
)
from models import CodeChunk, RepositoryFile


_PARSER_LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java",
    ".cpp": "cpp", ".cc": "cpp", ".c": "c", ".h": "cpp",
    ".hpp": "cpp", ".swift": "swift", ".go": "go", ".rs": "rust",
}

_CLASS_TYPES = {
    "class_definition", "class_declaration", "interface_declaration",
    "enum_declaration", "struct_item", "enum_item", "trait_item",
    "impl_item", "class_specifier", "struct_specifier",
}
_CALLABLE_TYPES = {
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "constructor_declaration", "function_item",
}
_IDENTIFIER_TYPES = {
    "identifier", "field_identifier", "property_identifier", "type_identifier",
}
@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    name: str
    kind: str
    parent_class: str | None = None


def _chunk_id(file: RepositoryFile, start: int, end: int, index: int) -> str:
    value = f"{file.repo_name}:{file.file_path}:{start}:{end}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _windows(file: RepositoryFile, lines: list[str], base_line: int, unit_name: str,
             unit_type: str, window_size: int, overlap: int,
             start_index: int = 0) -> list[CodeChunk]:
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError(
            "window_size must be positive and overlap must be between 0 and window_size - 1"
        )
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
    return _windows(file, file.content.splitlines(), 1, file.file_path,
                    "line_window", window_size, overlap)


def _node_lines(node: Node) -> tuple[int, int]:
    start_node = node.parent if node.parent and node.parent.type == "decorated_definition" else node
    return start_node.start_point.row + 1, start_node.end_point.row + 1


def _first_identifier(node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type in _IDENTIFIER_TYPES:
        return node.text.decode("utf-8", errors="replace")
    for child in node.named_children:
        value = _first_identifier(child)
        if value:
            return value
    return None


def _node_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", errors="replace")
    return _first_identifier(node.child_by_field_name("declarator") or node)


def _is_arrow_function_declaration(node: Node) -> bool:
    value = node.child_by_field_name("value")
    return node.type == "variable_declarator" and value is not None and value.type == "arrow_function"


def _callback_name(node: Node) -> str | None:
    """Name an anonymous arrow callback from its surrounding function call."""
    arguments = node.parent
    call = arguments.parent if arguments and arguments.type == "arguments" else None
    if call is None or call.type != "call_expression":
        return None
    function = call.child_by_field_name("function")
    if function is None:
        return None
    name = function.text.decode("utf-8", errors="replace")
    first_argument = next(
        (child for child in arguments.named_children if child is not node), None
    )
    if first_argument is not None and first_argument.type in {"string", "template_string"}:
        name += first_argument.text.decode("utf-8", errors="replace")
    return name


def _collect_units(root: Node) -> list[_Unit]:
    units: list[_Unit] = []

    def visit(node: Node, parent_class: str | None = None) -> None:
        if node.type in _CLASS_TYPES:
            name = _node_name(node)
            if name:
                start, end = _node_lines(node)
                units.append(_Unit(start, end, name, "class", parent_class))
                for child in node.named_children:
                    visit(child, name)
                return
        if node.type in _CALLABLE_TYPES or _is_arrow_function_declaration(node):
            name = _node_name(node)
            if name:
                start, end = _node_lines(node)
                qualified = f"{parent_class}.{name}" if parent_class else name
                is_async = any(child.type == "async" for child in node.children)
                kind = "method" if parent_class else "function"
                if is_async:
                    kind = f"async_{kind}"
                units.append(_Unit(start, end, qualified, kind, parent_class))
                return
        if node.type == "arrow_function":
            name = _callback_name(node)
            if name:
                start, end = _node_lines(node)
                kind = "async_function" if any(
                    child.type == "async" for child in node.children
                ) else "function"
                units.append(_Unit(start, end, name, kind, parent_class))
                return
        for child in node.named_children:
            visit(child, parent_class)

    visit(root)
    return units


def _remaining_ranges(lines: list[str], start: int, end: int,
                      excluded: set[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    range_start = last_nonblank = None
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


def chunk_tree_sitter(
    file: RepositoryFile,
    unit_window_size: int = TREE_SITTER_UNIT_WINDOW_SIZE,
    unit_overlap: int = TREE_SITTER_UNIT_OVERLAP,
) -> list[CodeChunk]:
    """Create symbol-aware chunks using the parser selected by file extension."""
    language = _PARSER_LANGUAGE_BY_EXTENSION.get(file.extension)
    if language is None:
        return chunk_generic(file)
    try:
        tree = get_parser(language).parse(file.content.encode("utf-8"))
    except Exception:
        return chunk_generic(file)
    if tree.root_node.has_error:
        return chunk_generic(file)

    lines = file.content.splitlines()
    units = _collect_units(tree.root_node)
    if not units:
        return chunk_generic(file)

    callable_units = [unit for unit in units if unit.kind != "class"]
    class_units = [unit for unit in units if unit.kind == "class"]
    ranges: list[tuple[int, int, str, str]] = [
        (unit.start, unit.end, unit.name, unit.kind) for unit in callable_units
    ]
    for unit in class_units:
        excluded = {
            line_no
            for child in callable_units
            if child.parent_class == unit.name
            for line_no in range(child.start, child.end + 1)
        }
        for start, end in _remaining_ranges(lines, unit.start, unit.end, excluded):
            ranges.append((start, end, unit.name, "class"))

    covered = {
        line_no
        for unit in units
        if unit.parent_class is None
        for line_no in range(unit.start, unit.end + 1)
    }
    for start, end in _remaining_ranges(lines, 1, len(lines), covered):
        ranges.append((start, end, file.file_path, "module"))

    chunks: list[CodeChunk] = []
    for start, end, name, kind in sorted(ranges):
        chunks.extend(_windows(
            file, lines[start - 1:end], start, name, kind,
            unit_window_size, unit_overlap, len(chunks),
        ))
    return chunks or chunk_generic(file)


def _python_node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def chunk_python(file: RepositoryFile, window_size: int = CHUNK_WINDOW_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> list[CodeChunk]:
    """Create Python chunks with the standard-library AST parser."""
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
            start = _python_node_start(node)
            end = node.end_lineno or node.lineno
            units.append((start, end, node.name, type_names[type(node)]))
            covered.update(range(start, end + 1))
        elif isinstance(node, ast.ClassDef):
            class_start = _python_node_start(node)
            class_end = node.end_lineno or node.lineno
            covered.update(range(class_start, class_end + 1))
            method_lines: set[int] = set()
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_start = _python_node_start(child)
                    method_end = child.end_lineno or child.lineno
                    method_type = "async_method" if isinstance(
                        child, ast.AsyncFunctionDef
                    ) else "method"
                    units.append((
                        method_start, method_end, f"{node.name}.{child.name}", method_type,
                    ))
                    method_lines.update(range(method_start, method_end + 1))
            for start, end in _remaining_ranges(
                lines, class_start, class_end, method_lines
            ):
                units.append((start, end, node.name, "class"))

    module_type = "module_script" if not units else "module"
    ordered = units + [
        (start, end, file.file_path, module_type)
        for start, end in _remaining_ranges(lines, 1, len(lines), covered)
    ]
    chunks: list[CodeChunk] = []
    for start, end, name, kind in sorted(ordered):
        chunks.extend(_windows(
            file, lines[start - 1:end], start, name, kind,
            window_size, overlap, len(chunks),
        ))
    return chunks


def chunk_file(file: RepositoryFile, window_size: int = CHUNK_WINDOW_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[CodeChunk]:
    if file.extension == ".py":
        return chunk_python(file, window_size, overlap)
    return chunk_tree_sitter(file)
