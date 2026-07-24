# Local RAG Codebase Assistant

A local-first Streamlit application that indexes source code and answers natural-language questions with answers grounded in retrieved excerpts. Repository content and vectors stay in the local process; only the selected excerpts and question are sent to Google Gemini for answer generation.

## Architecture

```text
Streamlit UI
    ↓
RAG Pipeline
    ├── Repository Loader
    ├── AST/Line-Based Chunker
    ├── Sentence Transformer Embedder
    ├── Qdrant Vector Store
    └── Gemini Answer Generator
```

```text
Repository → supported text files → logical chunks → normalized embeddings
           → in-memory Qdrant → question retrieval → Gemini → cited answer
```

## Features

- Recursively loads supported code and text files while ignoring dependency, build, editor, and hidden directories.
- Uses Python's `ast` module for top-level function, async-function, and class chunks.
- Indexes class methods as non-overlapping qualified symbols such as `Typer.command` while retaining separate class-overview chunks.
- Uses configurable line windows for other languages and oversized units.
- Embeds code locally with `sentence-transformers/all-MiniLM-L6-v2`.
- Isolates repositories with Qdrant payload filters and supports re-indexing.
- Generates context-only answers with the official `google-genai` SDK and citations such as `[src/auth/service.py, authenticate_user, lines 12-47]`.
- Shows retrieval scores and raw source chunks in the Streamlit chat UI.
- Retrieves a broad semantic candidate set, then reranks with exact identifiers, lexical overlap, implementation intent, file paths, and source diversity.

## Retrieval defaults

The defaults were selected using 98 known-answer questions across Flask, Typer,
and Requests:

| Parameter | Value |
|---|---:|
| Chunk window | 10 lines |
| Chunk overlap | 0 lines |
| Maximum returned chunks per file/symbol | 1 |
| Semantic candidates before reranking | At least 100 |
| Default answer context | 8 chunks |
| Qdrant distance | Cosine |

Python is first divided into AST functions, methods, and class regions. The
10-line window is only used when one of those units is larger than the window.
Zero overlap performed better than overlapping windows because shifted
duplicates competed for limited result positions. Smaller windows also avoid
silent truncation by MiniLM's 256-token input limit.

The reranker combines semantic similarity with code identifiers, symbol names,
filename tokens, general source/test/document priors, implementation intent,
and one-result-per-symbol diversity. Its weights are centralized in
`config.py`.

## Requirements and setup

Install Python 3.10 or newer, then run:

```bash
cd codebase-assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set your Gemini API key in `.env`:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

The key is read only when an answer is generated. Do not commit `.env`.

## Run

```bash
streamlit run app.py
```

Enter an absolute or relative local repository path in the sidebar, select **Index Repository**, and wait for indexing to finish. You can then ask questions such as:

- Where is user authentication implemented?
- What validation happens before an order is saved?
- How are configuration values loaded?
- Which function calls the payment provider?

Re-indexing a repository replaces only that repository's previous chunks. Indexes exist only for the life of the Streamlit Python process.

## Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit session state, indexing controls, chat, and source display |
| `repo_loader.py` | Safe recursive repository loading and skip accounting |
| `code_chunker.py` | Python AST-aware and generic line-window chunking |
| `embedder.py` | Lazy singleton embedding model and batch embedding |
| `vector_store.py` | In-memory Qdrant collection, filtering, search, and clearing |
| `llm_client.py` | All Gemini prompt construction and SDK calls |
| `rag_pipeline.py` | Public ingestion and question-answering orchestration |
| `models.py` | Shared data classes |
| `config.py` | Models, dimensions, limits, chunking, file, and directory settings |

Gemini's model is configured with `GEMINI_MODEL_NAME` in `config.py` and defaults to `gemini-3.5-flash`. This provider-specific integration is confined to `llm_client.py`, allowing a replacement provider to implement the same `generate_answer(question, retrieved_chunks)` boundary.

Transient Gemini capacity and rate-limit errors are retried. If the primary model remains unavailable, answer generation falls back to the stable `gemini-3.1-flash-lite` model.

## Testing

```bash
pytest -q
```

Tests cover repository filtering and relative paths, Python and generic chunk behavior, syntax-error fallback, large units, Qdrant insertion, repository filtering, and repository-scoped clearing. They do not call Gemini or download an embedding model.

## Retrieval and answer evaluation

`evaluate.py` ingests a repository and evaluates retrieval against a JSON
known-answer dataset. Reusable Flask, Requests, and Typer datasets are kept in
`evaluation/`; compact benchmark metrics are kept in `results/`.

Run retrieval-only evaluation without making Gemini API calls:

```bash
python evaluate.py /path/to/typer evaluation/evaluation_questions.example.json --top-k 8 --k-values 1,3,6,8 --output typer-evaluation.json
```

It reports useful Hit@k (primary or accepted alternative evidence), primary Hit@k, primary-source recall, chunk precision, and mean reciprocal rank at multiple cutoffs from the same retrieval run. The defaults are `k=1`, `k=3`, and `k=6`; customize them with `--k-values 1,3,6,10`. Every retrieved file, symbol, line range, and score is included in the JSON report. To include generated answers and empty manual 0–2 scoring fields, add `--generate`:

```bash
python evaluate.py /path/to/typer evaluation/evaluation_questions.example.json --top-k 8 --k-values 1,3,6,8 --generate --output typer-answer-evaluation.json
```

Run the retrieval-only command with several `--top-k` values (for example 3, 6, 8, and 12) and compare the reports. To test chunk-size or overlap changes, update `config.py`, rerun the command, and compare the same question set. Review and extend each question's `expected_sources` and `important_facts` as the ground truth evolves.

### Cross-repository optimization result

The optimized configuration was evaluated without Gemini calls:

| Repository | Questions | Useful Hit@6 | Primary Hit@6 | Recall@6 | Primary Hit@8 |
|---|---:|---:|---:|---:|---:|
| Flask | 30 | 86.7% | 83.3% | 73.1% | 93.3% |
| Typer | 40 | 90.0% | 60.0% | 52.1% | 60.0% |
| Requests | 28 | 89.3% | 85.7% | 82.1% | 96.4% |
| Weighted aggregate | 98 | 88.8% | 74.5% | 67.1% | 80.6% |

Typer's primary-source metric is lower than its useful-hit metric because many
questions retrieve valid accepted alternatives such as `TyperCommand`,
`get_click_param`, and rich help-formatting functions instead of every preferred
primary target.

## Limitations

- Structure-aware chunking is currently Python-specific; other languages use line-window chunking.
- Qdrant data is in memory and is lost after the application restarts.
- The application is intended primarily for local use and has no authentication or multi-user isolation.
- Retrieval may be less reliable for broad architectural questions whose evidence spans many files.
- The system performs semantic retrieval, not complete static program analysis.
- Source code in retrieved excerpts is sent to Gemini, so review the data policy appropriate to your repository.

## Future improvements

Potential extensions include persistent Qdrant storage, token-budget chunking,
language-aware parsers for more languages, sparse/dense hybrid retrieval,
repository identity based on full-path hashes, configurable ignore files,
neighbor-window expansion for long functions, and automated citation
validation.
