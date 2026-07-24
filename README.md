# Local RAG Codebase Assistant

A local Retrieval-Augmented Generation (RAG) application that indexes software repositories and answers natural-language questions about source code with file, symbol, and line-level citations.

Repository parsing, chunking, embedding generation, and retrieval are performed locally. Only the retrieved code excerpts and the user's question are sent to Google Gemini for answer generation.

---

## Features

- AST-aware code chunking for Python repositories
- Semantic code retrieval using MiniLM embeddings and Qdrant
- Grounded answer generation using Google Gemini
- Repository-aware retrieval heuristics to improve retrieval quality
- Retrieval evaluation using Hit@k, Recall@k, Precision@k, and MRR
- Automated unit tests

---

## Architecture

```text
Repository
    │
    ▼
Repository Loader
    │
    ▼
AST-aware Chunker
    │
    ▼
MiniLM Embeddings
    │
    ▼
Qdrant
    │
    ▼
Retriever
    │
    ▼
Google Gemini
    │
    ▼
Grounded Answer + Citations
```

---

## Tech Stack

- Python
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Qdrant
- Google Gemini API
- Pytest

---

## Installation

```bash
git clone https://github.com/nush20/codebase-assistant.git

cd codebase-assistant

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Questions

- Where is the `Session` class implemented?
- How does Flask dispatch requests?
- Where are Typer commands registered?
- What validation happens before an order is saved?
- Which function calls the payment provider?

---

## Evaluation

The retriever is evaluated using curated question sets across multiple open-source repositories.

Metrics:

- Hit@k
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)

### Current Results

| Repository | Questions | Useful Hit@6 | Useful Hit@8 |
|------------|----------:|-------------:|-------------:|
| Flask | 30 | 86.7% | 93.3% |
| Typer | 40 | 90.0% | 90.0% |
| Requests | 28 | 89.3% | 96.4% |
| Weighted aggregate | 98 | 88.8% | 92.9% |

---

## Project Structure

```text
codebase-assistant/
│
├── app.py
├── config.py
├── models.py
├── repo_loader.py
├── code_chunker.py
├── embedder.py
├── vector_store.py
├── rag_pipeline.py
├── llm_client.py
├── evaluate.py
│
├── evaluation/
├── tests/
│
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## Limitations

- AST-aware chunking currently supports Python only.
- Qdrant uses in-memory storage and must be rebuilt after restarting the application.
- Generated answers depend on the quality of retrieved context.

---

## Future Improvements

- Persistent vector storage
- Support for additional programming languages
- Hybrid retrieval
- Incremental indexing

---

## License

MIT License
