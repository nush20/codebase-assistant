"""Central configuration for the codebase assistant."""

import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32
CHUNK_WINDOW_SIZE = 10
CHUNK_OVERLAP = 0
MAX_FILE_SIZE_BYTES = 1_000_000
MAX_REPOSITORY_FILES = int(os.getenv("MAX_REPOSITORY_FILES", "2000"))
MAX_REPOSITORY_SIZE_BYTES = int(os.getenv("MAX_REPOSITORY_SIZE_BYTES", "50000000"))
DEFAULT_TOP_K = 8
RETRIEVAL_CANDIDATE_MULTIPLIER = 5
MIN_RETRIEVAL_CANDIDATES = 100
QDRANT_COLLECTION_NAME = "code_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_INFERENCE_MODEL = os.getenv(
    "QDRANT_INFERENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
IDENTIFIER_MATCH_WEIGHT = 0.35
TOKEN_OVERLAP_WEIGHT = 0.15
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
GEMINI_FALLBACK_MODEL_NAMES = ("gemini-3.1-flash-lite",)
GEMINI_MAX_ATTEMPTS_PER_MODEL = 2
GEMINI_RETRY_DELAY_SECONDS = 1.0
MAX_RESPONSE_TOKENS = 2048
API_BASE_URL = os.getenv("CODEBASE_ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "600"))

IGNORED_DIRECTORIES = frozenset({
    ".git", ".github", "node_modules", "venv", ".venv", "env",
    "__pycache__", "build", "dist", "coverage", ".idea", ".vscode",
    "target", "vendor",
})

IGNORED_FILES = frozenset({
    "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock",
    "bun.lock", "bun.lockb",
})

SUPPORTED_EXTENSIONS = frozenset({
    ".py", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".cc",
    ".c", ".h", ".hpp", ".swift", ".go", ".rs", ".html", ".css",
    ".scss", ".sass", ".md", ".txt",
    ".json", ".yaml", ".yml", ".toml",
})

SOURCE_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".cc",
    ".c", ".h", ".hpp", ".swift", ".go", ".rs", ".html", ".css",
    ".scss", ".sass",
})

LANGUAGE_BY_EXTENSION = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript Module",
    ".jsx": "JavaScript JSX",
    ".ts": "TypeScript", ".tsx": "TypeScript TSX", ".java": "Java",
    ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++ Header",
    ".hpp": "C++ Header", ".swift": "Swift", ".go": "Go", ".rs": "Rust",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".sass": "Sass",
    ".md": "Markdown", ".txt": "Text", ".json": "JSON", ".yaml": "YAML",
    ".yml": "YAML", ".toml": "TOML",
}
