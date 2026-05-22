"""
vector_store.py
All Qdrant interactions for RPaper.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()

# ── Embedding models ──────────────────────────────────────────────────────────
# FastEmbed is local; no API key needed.

from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384         # dimension of BAAI/bge-small-en-v1.5
CACHE_SIMILARITY_THRESHOLD = 0.95

dense_embeddings = FastEmbedEmbeddings(model_name=DENSE_MODEL)
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# ── Qdrant client ─────────────────────────────────────────────────────────────

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=120,
)

CACHE_COLLECTION = "rpaper_semantic_cache"


# ── Collection helpers ────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"rpaper_{session_id.replace('-', '_')}"


def _ensure_session_collection(collection_name: str) -> None:
    """Create the session collection with hybrid (dense + sparse) vectors if absent."""
    if qdrant_client.collection_exists(collection_name):
        return
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )


def _ensure_cache_collection() -> None:
    """Create the global semantic cache collection if absent."""
    if qdrant_client.collection_exists(CACHE_COLLECTION):
        return
    qdrant_client.create_collection(
        collection_name=CACHE_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
    )


# ── VectorStore factory ───────────────────────────────────────────────────────

def get_vectorstore(session_id: str) -> QdrantVectorStore:
    """Return a hybrid-search QdrantVectorStore for the given session."""
    collection_name = get_collection_name(session_id)
    _ensure_session_collection(collection_name)
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    """Embed and index child documents into the session's Qdrant collection."""
    store = get_vectorstore(session_id)
    store.add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    """Return deduplicated paper titles present in the session collection."""
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break

    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Hybrid similarity search against the session collection."""
    return get_vectorstore(session_id).similarity_search(query, k=k)


# ── Semantic cache ────────────────────────────────────────────────────────────

def check_cache(query: str, session_id: str) -> str | None:
    """
    Look for a cached answer to *query* within this session.

    Returns the cached answer string if similarity ≥ CACHE_SIMILARITY_THRESHOLD,
    otherwise returns None.
    """
    _ensure_cache_collection()

    query_vector = dense_embeddings.embed_query(query)

    results: list[ScoredPoint] = qdrant_client.search(
        collection_name=CACHE_COLLECTION,
        query_vector=("dense", query_vector),
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
        limit=1,
        with_payload=True,
    )

    if not results:
        return None

    top = results[0]
    if top.score >= CACHE_SIMILARITY_THRESHOLD:
        return (top.payload or {}).get("answer")

    return None


def save_to_cache(query: str, answer: str, session_id: str) -> None:
    """Embed *query* and store the question/answer pair in the semantic cache."""
    _ensure_cache_collection()

    query_vector = dense_embeddings.embed_query(query)
    point_id = __import__("uuid").uuid4().hex  # unique str-compatible ID

    qdrant_client.upsert(
        collection_name=CACHE_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector={"dense": query_vector},
                payload={"question": query, "answer": answer, "session_id": session_id},
            )
        ],
    )