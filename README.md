# Agentic RAG System for Academic Research

A production-grade Retrieval-Augmented Generation (RAG) backend designed specifically for interacting with, querying, and fact-checking complex academic research papers. This system leverages a multi-agent **LangGraph** architecture, hybrid search, hierarchical retrieval, and strict hallucination prevention mechanisms to deliver highly accurate, verifiable answers.

## 🛠️ Core Technologies
*   **Orchestration:** LangGraph, LangChain, FastAPI
*   **Vector Database:** Qdrant (Local Docker Deployment)
*   **Relational Database:** PostgreSQL (Parent Documents & Chat History)
*   **Embeddings:** FastEmbed (BAAI/bge-small-en-v1.5 & prithivida/Splade_PP_en_v1)
*   **Reranker:** FlashRank
*   **Large Language Models:** OpenAI (GPT-4o, GPT-4o-mini)
*   **Web Search:** Tavily API
*   **Evaluation:** Ragas, HuggingFace Datasets

---

## 🏗️ End-to-End Architecture

The system is broken down into two distinct phases: Document Ingestion and Query Execution.

### Phase 1: Document Ingestion
When a user uploads a PDF research paper, the system ensures optimal retrieval performance without wasting compute resources.

1.  **Hierarchical Chunking (Parent-Child):** The document is split into large context blocks ("Parent" chunks) and smaller semantic blocks ("Child" chunks). 
2.  **Embedding Caching (SQLite):** Before generating vectors, the system computes a SHA-256 hash of each text chunk. It checks a local SQLite database (`embedding_cache.db`). If the hash exists, it instantly loads the pre-computed vectors. If it is a new chunk, it runs through the local Dense and Sparse embedding models and caches the result.
3.  **Storage:** The small "Child" chunks and their vectors are saved to Qdrant for fast semantic searching. The large "Parent" chunks are saved to PostgreSQL.

### Phase 2: The Journey of a User Query
When a user submits a question, the query navigates a deterministic, directed acyclic graph (DAG) managed by LangGraph.

#### Step 1: Semantic Caching
The query is immediately embedded and checked against a dedicated Qdrant cache collection (`rpaper_semantic_cache_v3`). If a mathematically similar query (Cosine Similarity > 0.90) was previously answered, the system halts execution and instantly returns the cached answer, saving API costs and reducing latency to milliseconds.

#### Step 2: Intelligent Routing
If it is a cache miss, an LLM Router evaluates the user's prompt to determine their intent. It directs the graph down one of two paths: **Question Answering (QA)** or **Claim Verification**.

#### Step 3A: The QA Pipeline
If the user is asking a general question about the paper:
1.  **Query Rewrite & Expansion:** An LLM analyzes the chat history to rewrite the input into a standalone query. It also generates multiple synonymous queries to increase search recall. 
2.  **Web Search Detection:** During the rewrite, if the LLM detects the user is asking about current events or external knowledge not found in the paper, it flips a `needs_web_search` flag.
3.  **Hybrid Retrieval:** The system performs a Reciprocal Rank Fusion (RRF) search on Qdrant, using both dense vectors (semantic meaning) and sparse vectors (exact keyword matching). If the web flag is true, it concurrently searches the internet via Tavily and adds live web pages to the retrieval pool.
4.  **Hierarchical Context Assembly:** For every small chunk retrieved from Qdrant, the system queries PostgreSQL to fetch the massive, surrounding "Parent" chunk.
5.  **Cross-Encoder Reranking:** The assembled context blocks are passed through FlashRank. This local cross-encoder model mathematically scores every paragraph against the user's query simultaneously, floating the absolute most relevant information to the top.
6.  **Relevance Gate:** An LLM acts as a strict filter. If the top-ranked documents do not contain the answer, it triggers a conversational fallback to prevent hallucinations.
7.  **Grounding Verification:** After generating the answer, an impartial "Judge" LLM verifies that *every single factual claim* in the generated text is explicitly supported by the retrieved context. If it detects a hallucination, it silently deletes the answer and forces the generator to retry.

#### Step 3B: The Claim Verification Pipeline
If the user explicitly asks to fact-check a claim:
1.  **Claim Extraction:** The system breaks the user's prompt down into individual, discrete factual statements.
2.  **Dual Evidence Collection:** The system concurrently searches the local uploaded academic papers and live scientific internet domains (via Tavily) for evidence supporting or refuting the extracted claims.
3.  **Verdict Generation:** The system synthesizes the internal and external evidence to produce a strict factual verdict.

---

## 📊 Automated Evaluation Suite
The repository includes a production-grade evaluation pipeline (`evaluation/evaluate_rag.py`) utilizing the **Ragas** framework. 

The `generate_dataset.py` script acts autonomously, pulling complex paragraphs from Qdrant and using GPT-4o-mini to generate synthetic test questions and ground-truth answers. The RAG pipeline is then rigorously graded against this dataset.

**Current Pipeline Performance (Sample Size: 25 Complex Q&A Pairs):**

| Metric | Average Score | Description |
| :--- | :--- | :--- |
| **Context Precision** | `0.907` | Measures the proportion of relevant chunks correctly ranked at the top by the FlashRank Cross-Encoder. |
| **Faithfulness** | `0.775` | Measures the factual consistency of the generated answer against the retrieved context (success of the Grounding Check node). |
| **Answer Relevancy** | `0.666` | Measures how directly the generated answer addresses the user's original query. |

---

## ⚙️ Setup and Execution

1. **Start the Databases**
Ensure Docker is running, then spin up the Qdrant and PostgreSQL containers:
```bash
docker-compose up -d
```

2. **Start the Backend**
Launch the FastAPI server:
```bash
fastapi dev main.py
```

3. **Run the Ragas Evaluation (Optional)**
Generate a synthetic dataset and calculate pipeline metrics:
```bash
python -m evaluation.generate_dataset
python -m evaluation.evaluate_rag
```
