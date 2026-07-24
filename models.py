"""Data models shared by the ingestion and question-answering pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepositoryFile:
    repo_name: str
    file_path: str
    extension: str
    language: str
    content: str


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    repo_name: str
    file_path: str
    language: str
    text: str
    start_line: int
    end_line: int
    unit_name: str
    unit_type: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: CodeChunk
    score: float


@dataclass
class IngestionResult:
    repo_name: str
    files_processed: int
    chunks_created: int
    skipped_file_count: int
    indexing_errors: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    answer: str
    retrieved_sources: list[RetrievedChunk] = field(default_factory=list)
