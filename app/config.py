from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    index_path: Path
    api_key: str
    model: str
    embedding_model: str
    retriever_k: int
    temperature: float
    max_tokens: int
    bm25_weight: float
    vector_weight: float
    rerank_top_n: int

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        return cls(
            project_root=project_root,
            index_path=project_root / "faiss_index",
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            retriever_k=int(os.getenv("RETRIEVER_K", "4")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.6")),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.4")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "10")),
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def validate(self) -> None:
        if not self.has_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file or environment.")

        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index folder not found: {self.index_path}")

        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
