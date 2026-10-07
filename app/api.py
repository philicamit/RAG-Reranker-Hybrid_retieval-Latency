from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .rag_pipeline import RAGService


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the RAG application.")
    k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve.")


class AskResponse(BaseModel):
    answer: str
    context: list[str]
    latency: dict[str, float]


@lru_cache(maxsize=1)
def get_api_service() -> RAGService:
    settings = Settings.from_env()
    settings.validate()
    return RAGService(settings)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Game of Thrones QA API",
        version="1.0.0",
        description="Production-ready API for the Game of Thrones RAG application.",
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Game of Thrones QA API is running."}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            service = get_api_service()
            answer, docs, metrics = service.answer_with_context(request.question)
            return AskResponse(
                answer=answer,
                context=[doc.page_content for doc in docs],
                latency=metrics.to_dict(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
