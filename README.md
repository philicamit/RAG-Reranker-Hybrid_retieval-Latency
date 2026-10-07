# 🐺 Game of Thrones Book QA System

A production-style Retrieval-Augmented Generation (RAG) application built for answering questions from a local Game of Thrones text corpus using a hybrid search stack, reranking, and latency-aware retrieval and generation monitoring.

This project combines:

* Streamlit for interactive user experience
* FastAPI for backend API access
* FAISS for vector retrieval
* BM25 for lexical retrieval
* OpenAI embeddings and GPT models for semantic matching and answer generation
* Application-level latency tracking for TTFT, retrieval time, rerank time, and end-to-end response time

---

## 1. Why this application exists

The core purpose is to answer questions grounded in a domain-specific knowledge base while minimizing hallucination and improving answer quality. Instead of relying only on the language model to remember everything, the system first retrieves the most relevant chunks from a local document index and then asks the LLM to answer using only that context.

This improves:

* factual grounding
* answer relevance
* context-aware reasoning
* control over hallucination risk

---

## 2. High-level architecture

The system has three major layers:

1. Data and index layer
   - vector index is stored in `faiss_index/`
   - document content is loaded from the existing persisted FAISS store

2. Retrieval and ranking layer
   - BM25 lexical retrieval finds text matches by token overlap
   - HNSW vector retrieval finds semantically similar chunks
   - weighted hybrid scoring combines both signals
   - reranking improves final ordering before generation

3. Generation and interface layer
   - LLM generates the final answer using the retrieved context
   - Streamlit provides the user interface
   - FastAPI exposes the same logic as a programmatic API

---

## 3. Workflow of the application

### Step 1: Load configuration
The app reads settings from environment variables and `.env` via `app/config.py`.

This includes values such as:

* `OPENAI_API_KEY`
* `OPENAI_MODEL`
* `OPENAI_EMBEDDING_MODEL`
* `RETRIEVER_K`
* `BM25_WEIGHT`
* `VECTOR_WEIGHT`
* `RERANK_TOP_N`

Why this matters:

* keeps credentials and runtime logic separate from code
* makes deployment cleaner and safer
* enables environment-specific tuning without changing source files

### Step 2: Initialize the retrieval service
`app/rag_pipeline.py` loads the FAISS vectorstore and builds the retrieval pipeline.

It does the following:

* loads the stored vector index from disk
* loads document text from the FAISS docstore
* creates embeddings for the corpus
* builds a BM25 index for lexical retrieval
* builds an HNSW FAISS index for semantic retrieval

Why this matters:

* static indexing avoids reindexing on every query
* better performance for repeated use
* faster answer generation in production-like scenarios

### Step 3: Query retrieval
When a user asks a question:

1. BM25 retrieves candidate chunks using term overlap
2. HNSW vector search retrieves semantically similar content
3. Both result sets are normalized and combined with weighted scoring
4. The combined candidates are reranked
5. The top results are passed to the LLM

This is a hybrid retrieval approach.

Why hybrid retrieval is used:

* BM25 handles exact keyword matching well
* vector search handles semantic phrasing well
* combining them gives stronger recall and better ranking quality

---

## 4. Reranking logic

The system uses a weighted hybrid score before sending context into the LLM.

The idea is:

* lexical signals and embedding signals are both valuable
* certain queries are more keyword-heavy
* others are more semantic or conversational
* a combined strategy reduces ranking failure cases

The blending is controlled by:

* `BM25_WEIGHT` (default: `0.6`)
* `VECTOR_WEIGHT` (default: `0.4`)

This means lexical retrieval is weighted slightly more heavily by default, but semantic similarity still contributes strongly.

Why rerank is important:

* the first-pass candidate set may contain noisy matches
* reranking ensures the model sees the most relevant context first
* improves context quality and answer accuracy

---

## 5. Latency monitoring and TTFT

The app measures several performance metrics:

* retrieval time
* rerank time
* generation time
* TTFT (Time To First Token)
* total end-to-end latency

This is implemented in `app/rag_pipeline.py` using Python timing around retrieval, reranking, LLM invocation, and total completion.

Why latency matters:

* helps diagnose whether the bottleneck is vector search, rerank, or model generation
* enables tuning for user-facing responsiveness
* supports production benchmarking and optimization

Common metrics tracked:

* TTFT = how long before the model starts returning output
* retrieval latency = how long search takes
* total latency = how long the end-to-end request takes

---

## 6. Streamlit frontend

The frontend is implemented in `app/ui.py`.

It provides:

* a chat-based interface
* a question box for end users
* answer display
* retrieved context expansion
* latency metric cards for TTFT, retrieval, reranking, and total time

Why this is useful:

* simple for demo and internal use
* easy for non-technical users to test the app
* allows quick evaluation of answer quality and performance

---

## 7. FastAPI backend

The backend is implemented in `app/api.py`.

It exposes:

* `/` — app health and service metadata
* `/health` — liveness endpoint
* `/ask` — question-answer API endpoint

The API request model is:

```json
{
  "question": "Who is Daenerys Targaryen?",
  "k": 4
}
```

The API response includes:

```json
{
  "answer": "...",
  "context": ["..."],
  "latency": {
    "retrieval_ms": 12.3,
    "rerank_ms": 4.8,
    "generation_ms": 850.2,
    "ttft_ms": 900.1,
    "total_ms": 1100.6
  }
}
```

Why FastAPI is included:

* separate backend service for production deployment
* API-first integrations for apps, services, or dashboards
* decouples UI from underlying retrieval logic

---

## 8. Files and responsibilities

* `main.py` — application entry point for Streamlit
* `api.py` — FastAPI server entry point
* `app/config.py` — environment and configuration
* `app/rag_pipeline.py` — retrieval logic, hybrid search, reranking, latency tracking, LLM invocation
* `app/ui.py` — Streamlit interface
* `app/api.py` — FastAPI REST API
* `faiss_index/` — local vector index and persisted data
* `requirements.txt` — project dependencies
* `tests/test_api.py` — API regression test

---

## 9. Why the architecture is production-friendly

This project is designed to be more than a notebook-style demo.

It includes:

* modular file structure
* environment-based config
* validation for missing secrets and missing indexes
* service layer separation
* API and UI both supported
* retrieval monitoring and latency metrics
* hybrid search rather than single-method retrieval

This makes it easier to:

* debug behavior
* improve retrieval quality
* benchmark latency
* deploy in a real environment
* scale into a team workflow

---

## 10. Running the application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add environment variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_openai_key_here
```

Optional tuning variables:

```bash
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RETRIEVER_K=4
BM25_WEIGHT=0.6
VECTOR_WEIGHT=0.4
RERANK_TOP_N=10
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=500
```

### Start the backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Start the frontend

```bash
streamlit run main.py
```

---

## 11. System value and production outlook

This application demonstrates a practical RAG system for document-grounded question answering, with a stronger retrieval stack than a simple vector-only approach. It combines lexical and semantic retrieval, uses reranking to improve the quality of candidate context, and exposes latency metrics so the system can be evaluated and optimized for real-world use.

It is designed to support:

* domain-specific knowledge assistants
* enterprise document search experiences
* production-quality RAG experimentation
* scalable question-answering workflows built on top of a local corpus

---

## 12. Summary

This project brings together search, ranking, generation, and observability in a single application. It uses hybrid retrieval to balance exact matching and semantic understanding, incorporates reranking to surface the most relevant evidence, and measures latency across the retrieval pipeline to support performance tuning and monitoring.

The result is a more reliable and transparent question-answering system than a standard FAISS-only retrieval flow, while still remaining easy to run, extend, and integrate into downstream applications.