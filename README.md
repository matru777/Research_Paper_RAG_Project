# RAG System for Academic Research

A production-grade Retrieval-Augmented Generation (RAG) backend designed for querying and fact-checking academic research papers. It utilizes a deterministic **LangGraph** orchestration pipeline to coordinate hybrid search, hierarchical retrieval, and strict hallucination prevention.

## Core Architecture & Features

### 1. Document Ingestion
*   **Asynchronous Processing:** PDF uploads are processed in the background via a **Redis + ARQ** worker queue to ensure a non-blocking API.
*   **Markdown Extraction:** Uses `pymupdf4llm` to convert PDFs natively into structured Markdown, preserving academic tables, lists, and headers.
*   **Hierarchical Chunking:** Splits text into large context blocks ("Parent" chunks stored in PostgreSQL) and small semantic snippets ("Child" chunks stored in Qdrant).
*   **Embedding Cache:** A local SQLite database hashes text chunks to bypass redundant CPU inference for BGE Dense and SPLADE Sparse embedding models.

### 2. Query Pipeline (LangGraph)
*   **Semantic Cache:** An incoming query is checked against Qdrant (`rpaper_semantic_cache_v3`). If similarity exceeds 0.90, the pipeline instantly returns the cached answer.
*   **Intent Routing:** An LLM router directs queries to either a standard QA pipeline or a strict Claim Verification pipeline.
*   **Hybrid Retrieval:** Queries execute a Reciprocal Rank Fusion (RRF) search across both dense semantic vectors and sparse exact-keyword vectors.
*   **Web Search Fallback:** If the LLM determines the query requires external world knowledge, it searches the live internet via **Tavily** and appends the results.
*   **Cross-Encoder Reranking:** The retrieved documents pass through **FlashRank**, mathematically scoring and ranking the most precise paragraphs at the absolute top.
*   **Strict Grounding Check:** After answer generation, a final Judge LLM verifies that every factual claim in the output is strictly supported by the context. If a hallucination is detected, the pipeline forces a retry.

## Automated Evaluation Suite
The system includes an automated evaluation pipeline (`evaluation/evaluate_rag.py`) utilizing the **Ragas** framework and synthetic datasets.

**Current Performance (25 Synthetic Q&A Pairs):**

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Context Precision** | `0.907` | Accuracy of FlashRank retrieving the correct chunks. |
| **Faithfulness** | `0.775` | Factual consistency of the generated answer against the context. |
| **Answer Relevancy** | `0.666` | Directness of the generated answer to the user's prompt. |

## Tech Stack
*   **Backend:** FastAPI, Python, LangGraph, ARQ (Redis)
*   **Databases:** Qdrant (Vector), PostgreSQL (Relational), SQLite (Cache)
*   **AI Models:** FastEmbed (BGE + SPLADE), FlashRank, OpenAI (GPT-4o)

## Setup & Execution
```bash
# 1. Start Vector and Relational Databases
docker-compose up -d

# 2. Start Backend Server
fastapi dev main.py

# 3. Run Evaluation (Optional)
python -m evaluation.generate_dataset
python -m evaluation.evaluate_rag
```
