"""Domain errors exposed by the RAG service boundary."""


class RepositoryNotIndexedError(ValueError):
    """Raised when a query targets a repository with no indexed chunks."""


class IndexingServiceError(RuntimeError):
    """Raised when an indexing dependency fails."""


class RetrievalServiceError(RuntimeError):
    """Raised when an embedding or vector-search dependency fails."""


class GenerationServiceError(RuntimeError):
    """Raised when answer generation fails."""
