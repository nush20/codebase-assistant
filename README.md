# Codebase Assistant

A RAG-based application that lets users ask questions about a public GitHub repository and receive answers grounded in its source code.

**Live demo:** [codebase-assistant-j8fnncauv7nhbhm4kk6vtj.streamlit.app](https://codebase-assistant-j8fnncauv7nhbhm4kk6vtj.streamlit.app/?embed=true)

## Features

- Loads local repositories or public GitHub URLs
- Uses Python AST parsing to create function, class, and method chunks
- Creates embeddings with `all-MiniLM-L6-v2`
- Stores and searches embeddings with Qdrant
- Improves ranking with identifier matching and token overlap
- Adds adjacent chunks when a function spans multiple chunks
- Generates grounded answers with Gemini
- Shows source files, symbols, and line numbers
- Provides Streamlit and FastAPI interfaces

## How It Works

```text
GitHub repository
      ↓
Load and chunk source files
      ↓
Create embeddings
      ↓
Store and search in Qdrant
      ↓
Rank the relevant code chunks
      ↓
Send context to Gemini
      ↓
Answer with source citations
```

## Tech Stack

- Python
- FastAPI
- Streamlit
- Qdrant
- Sentence Transformers
- Google Gemini
- Pytest

## Run Locally

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/nush20/codebase-assistant.git
cd codebase-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Add your Gemini and Qdrant credentials to `.env`.

Start the backend:

```bash
uvicorn api:app --reload
```

In a second terminal, start the frontend:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Example Repositories

### micrograd

```text
https://github.com/karpathy/micrograd.git
```

Example questions:

- How does `Value.backward()` calculate gradients?
- How does `Value.__mul__()` propagate gradients?
- How does a neuron calculate its output?
- How does the MLP construct its layers?

### schedule

```text
https://github.com/dbader/schedule.git
```

Example questions:

- How does `Scheduler.every()` create a job?
- How does `Scheduler.run_pending()` select jobs?
- How does `Job.do()` register a function?
- How does `Job.run()` execute and reschedule a job?

## Retrieval Evaluation

The current retrieval configuration was evaluated using curated questions and `top_k=8`.

| Repository | Questions | Hit@6 | Hit@8 | Recall@8 | Precision@8 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Flask | 30 | 80.0% | 83.3% | 67.2% | 43.8% | 0.629 |
| Typer | 20 | 45.0% | 55.0% | 22.5% | 12.5% | 0.328 |
| Requests | 28 | 92.9% | 96.4% | 84.8% | 45.5% | 0.647 |
| **Weighted total** | **78** | **75.6%** | **80.8%** | **62.1%** | **36.4%** | **0.558** |

These metrics measure retrieval quality only. Example answers in the live demo are generated separately by Gemini using the retrieved source chunks.

## Testing

Run the automated tests with:

```bash
pytest -q
```

Retrieval can also be evaluated using Hit@K, Recall@K, Precision@K, and Mean Reciprocal Rank.

## Limitations

- Python receives structure-aware AST chunking; other languages use line-based chunks.
- Only public GitHub repositories can be cloned by the deployed application.
- Answer quality depends on the retrieved source context and the Gemini response.
- Free hosting may require a short cold-start wait.

## License

This project is licensed under the MIT License.
