"""Thin FastAPI adapter for the codebase assistant RAG pipeline."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from config import DEFAULT_TOP_K
from errors import (
    GenerationServiceError,
    IndexingServiceError,
    RepositoryNotIndexedError,
    RetrievalServiceError,
)
from rag_pipeline import answer_question, ingest_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexRepositoryRequest(APIModel):
    repo_path: NonEmptyString


class IndexRepositoryResponse(APIModel):
    status: str
    repo_name: str
    files_processed: int
    chunks_created: int
    skipped_file_count: int
    indexing_errors: list[str]


class QueryRequest(APIModel):
    repo_name: NonEmptyString
    question: NonEmptyString
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class SourceResponse(APIModel):
    file: str
    symbol: str
    line_start: int
    line_end: int
    language: str
    unit_type: str
    text: str
    score: float


class QueryResponse(APIModel):
    answer: str
    sources: list[SourceResponse]


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Codebase Assistant API startup")
    yield
    logger.info("Codebase Assistant API shutdown")


app = FastAPI(
    title="Codebase Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/repositories/index", response_model=IndexRepositoryResponse)
def index_repository(request: IndexRepositoryRequest) -> IndexRepositoryResponse:
    try:
        result = ingest_repo(request.repo_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexingServiceError as exc:
        raise HTTPException(status_code=503, detail="Repository indexing is unavailable.") from exc
    except Exception as exc:
        logger.exception("Unexpected repository indexing failure")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
    return IndexRepositoryResponse(
        status="indexed",
        repo_name=result.repo_name,
        files_processed=result.files_processed,
        chunks_created=result.chunks_created,
        skipped_file_count=result.skipped_file_count,
        indexing_errors=result.indexing_errors,
    )


@app.post("/query", response_model=QueryResponse)
def query_repository(request: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(request.repo_name, request.question, request.top_k)
    except RepositoryNotIndexedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RetrievalServiceError, GenerationServiceError) as exc:
        raise HTTPException(status_code=502, detail="A RAG dependency is unavailable.") from exc
    except Exception as exc:
        logger.exception("Unexpected query failure")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                file=item.chunk.file_path,
                symbol=item.chunk.unit_name,
                line_start=item.chunk.start_line,
                line_end=item.chunk.end_line,
                language=item.chunk.language,
                unit_type=item.chunk.unit_type,
                text=item.chunk.text,
                score=item.score,
            )
            for item in result.retrieved_sources
        ],
    )
