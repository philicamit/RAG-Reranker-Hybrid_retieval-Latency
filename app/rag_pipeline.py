from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    ttft_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class RAGService:
    def __init__(self, settings: Settings):
        settings.validate()

        self.settings = settings
        self.embedding_model = OpenAIEmbeddings(model=settings.embedding_model)
        self.vectorstore = FAISS.load_local(
            str(settings.index_path),
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

        self.documents = self._load_documents()
        self.doc_texts = [doc.page_content for doc in self.documents]
        self.doc_embeddings = self._load_doc_embeddings(self.doc_texts)
        self.bm25 = BM25Okapi([text.lower().split() for text in self.doc_texts]) if self.doc_texts else None
        self.hnsw_index = self._build_hnsw_index()
        self.prompt = PromptTemplate(
            template="""
You are a helpful assistant that answers questions only based on the provided context.
If the answer is not contained within the context, respond with "I don't know. Please try asking another question."

Context:
{context}

Question: {question}
            """,
            input_variables=["context", "question"],
        )
        self.llm = ChatOpenAI(
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def _load_documents(self) -> list[Document]:
        doc_ids = list(self.vectorstore.index_to_docstore_id.values())
        documents = []
        for doc_id in doc_ids:
            doc = self.vectorstore.docstore.search(doc_id)
            if doc is not None:
                documents.append(doc)
        return documents

    def _load_doc_embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        embeddings = self.embedding_model.embed_documents(texts)
        return np.asarray(embeddings, dtype="float32")

    def _build_hnsw_index(self):
        if self.doc_embeddings.size == 0:
            return None

        dim = self.doc_embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        index.add(self.doc_embeddings)
        return index

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _query_hnsw(self, question: str, k: int) -> list[tuple[int, float]]:
        if self.hnsw_index is None:
            return []

        query_vector = np.asarray(self.embedding_model.embed_query(question), dtype="float32").reshape(1, -1)
        distances, indices = self.hnsw_index.search(query_vector, min(k, len(self.documents)))

        results: list[tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            similarity = 1.0 / (1.0 + float(dist))
            results.append((int(idx), similarity))
        return results

    def _query_bm25(self, question: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []

        tokenized_query = question.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        return [(idx, float(score)) for idx, score in ranked]

    def _rerank_hybrid(self, bm25_hits: list[tuple[int, float]], vector_hits: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        combined: dict[int, float] = {}
        bm25_max = max((score for _, score in bm25_hits), default=1.0)
        vector_max = max((score for _, score in vector_hits), default=1.0)

        for idx, score in bm25_hits:
            combined[idx] = combined.get(idx, 0.0) + (self.settings.bm25_weight * (score / bm25_max if bm25_max else 0.0))

        for idx, score in vector_hits:
            combined[idx] = combined.get(idx, 0.0) + (self.settings.vector_weight * (score / vector_max if vector_max else 0.0))

        reranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        return [(idx, score) for idx, score in reranked[:k]]

    def get_relevant_context(self, question: str, k: int | None = None) -> list[Document]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        docs_to_fetch = k or self.settings.retriever_k
        search_top_k = max(docs_to_fetch * 3, self.settings.rerank_top_n)

        bm25_hits = self._query_bm25(question, search_top_k)
        vector_hits = self._query_hnsw(question, search_top_k)
        ranked_indices = self._rerank_hybrid(bm25_hits, vector_hits, docs_to_fetch)

        ranked_docs: list[Document] = []
        for idx, _ in ranked_indices:
            if 0 <= idx < len(self.documents):
                ranked_docs.append(self.documents[idx])

        if len(ranked_docs) < docs_to_fetch:
            fallback_docs = self.vectorstore.similarity_search(question, k=docs_to_fetch)
            for doc in fallback_docs:
                if doc not in ranked_docs:
                    ranked_docs.append(doc)
                    if len(ranked_docs) >= docs_to_fetch:
                        break

        return ranked_docs[:docs_to_fetch]

    def answer(self, question: str) -> tuple[str, list[Document], LatencyMetrics]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        docs = self.get_relevant_context(question)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        rerank_start = time.perf_counter()
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        context = self._format_docs(docs)
        prompt = self.prompt.format(context=context, question=question)
        generation_start = time.perf_counter()
        response = self.llm.invoke(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0

        ttft_ms = (generation_start - total_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        metrics = LatencyMetrics(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        )
        logger.info("RAG answer generated with latency metrics: %s", metrics.to_dict())
        return response.content, docs, metrics

    def answer_with_context(self, question: str) -> tuple[str, list[Document], LatencyMetrics]:
        return self.answer(question)
