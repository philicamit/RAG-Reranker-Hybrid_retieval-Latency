from __future__ import annotations

import streamlit as st

from .config import Settings
from .rag_pipeline import RAGService


@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_rag_service() -> RAGService:
    settings = Settings.from_env()
    settings.validate()
    return RAGService(settings)


def _render_sidebar() -> None:
    st.sidebar.header("System settings")
    settings = Settings.from_env()
    st.sidebar.caption("Current configuration")
    st.sidebar.json(
        {
            "model": settings.model,
            "embedding_model": settings.embedding_model,
            "retriever_k": settings.retriever_k,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "bm25_weight": settings.bm25_weight,
            "vector_weight": settings.vector_weight,
            "rerank_top_n": settings.rerank_top_n,
        }
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "This app uses hybrid retrieval (BM25 + HNSW), reranking, and latency instrumentation for production-style search quality monitoring."
    )


def render_app() -> None:
    st.set_page_config(page_title="Game of Thrones QA", page_icon="📚", layout="wide")
    st.title("Game of Thrones Book QA")
    st.caption("Production-grade RAG application with hybrid retrieval, reranking, and latency tracking.")

    _render_sidebar()

    try:
        settings = Settings.from_env()
        if not settings.has_api_key:
            st.warning("OPENAI_API_KEY is missing. Add it to your .env file or environment variables.")
            return

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask a question about the book...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                service = get_rag_service()
                with st.spinner("Searching the corpus and generating the answer..."):
                    answer, docs, metrics = service.answer_with_context(prompt)

                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("TTFT", f"{metrics.ttft_ms:.0f} ms")
                col2.metric("Retrieval", f"{metrics.retrieval_ms:.0f} ms")
                col3.metric("Rerank", f"{metrics.rerank_ms:.0f} ms")
                col4.metric("Total", f"{metrics.total_ms:.0f} ms")

                with st.expander("View retrieved context"):
                    for idx, doc in enumerate(docs, start=1):
                        st.markdown(f"### Match {idx}")
                        st.write(doc.page_content)

                with st.expander("Latency details"):
                    st.json(metrics.to_dict())

            except Exception as exc:  # pragma: no cover - UI error path
                with st.chat_message("assistant"):
                    st.error(f"Something went wrong: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {exc}"})

        if not st.session_state.messages:
            st.info("Type a question in the chat box to start.")

    except Exception as exc:  # pragma: no cover - UI bootstrap error path
        st.error(f"Application configuration error: {exc}")
