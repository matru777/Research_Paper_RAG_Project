import uuid
import hashlib
import sqlite3
import json
from typing import Optional
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from qdrant_client.http import models as qdrant_models
from core.config import Settings
from fastembed import TextEmbedding, SparseTextEmbedding
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

cache_conn = sqlite3.connect("embedding_cache.db", check_same_thread=False)
cache_conn.execute("CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, dense TEXT, sparse TEXT)")
cache_conn.commit()

def get_hash(text: str) -> str:
    """Creates a SHA256 hash of the text chunk for caching."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cached_embeddings(texts: list[str]):
    """Fetches embeddings from SQLite cache or computes them if missing."""
    dense_out = [None] * len(texts)
    sparse_out = [None] * len(texts)
    to_compute_idx = []
    
    for i, text in enumerate(texts):
        h = get_hash(text)
        row = cache_conn.execute("SELECT dense, sparse FROM embeddings WHERE hash = ?", (h,)).fetchone()
        if row:
            # Cache Hit!
            dense_out[i] = json.loads(row[0])
            sparse_dict = json.loads(row[1])
            sparse_out[i] = qdrant_models.SparseVector(
                indices=sparse_dict["indices"],
                values=sparse_dict["values"]
            )
        else:
            # Cache Miss
            to_compute_idx.append(i)
            
    if to_compute_idx:
        texts_to_compute = [texts[i] for i in to_compute_idx]
        new_dense = list(dense_model.embed(texts_to_compute, batch_size=16))
        new_sparse = list(sparse_model.embed(texts_to_compute, batch_size=16))
        
        for i, dense_v, sparse_v in zip(to_compute_idx, new_dense, new_sparse):
            dense_list = dense_v.tolist()
            sparse_dict = {"indices": sparse_v.indices.tolist(), "values": sparse_v.values.tolist()}
            
            dense_out[i] = dense_list
            sparse_out[i] = qdrant_models.SparseVector(
                indices=sparse_dict["indices"],
                values=sparse_dict["values"]
            )
            
            # Save to SQLite
            h = get_hash(texts[i])
            cache_conn.execute(
                "INSERT OR REPLACE INTO embeddings (hash, dense, sparse) VALUES (?, ?, ?)",
                (h, json.dumps(dense_list), json.dumps(sparse_dict))
            )
        cache_conn.commit()
        
    return dense_out, sparse_out

# --- Qdrant Setup ---
_api_key = Settings.QDRANT_API_KEY if Settings.QDRANT_API_KEY else None
qdrant_client = QdrantClient(
    url=Settings.QDRANT_URL,
    api_key=_api_key,
    timeout=60
)
MAIN_COLLECTION = "research_papers_v2"
CACHE_COLLECTION = "rpaper_semantic_cache_v3"
CACHE_THRESHOLD = 0.90 

def ensure_collection_exists(collection_name: str, vector_size: int = 384):
    """Creates the Qdrant collection if it doesn't exist."""
    if not qdrant_client.collection_exists(collection_name):
        if collection_name == MAIN_COLLECTION:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": qdrant_models.SparseVectorParams(
                        index=qdrant_models.SparseIndexParams(on_disk=False)
                    )
                }
            )
        else:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="session_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

def add_paper(docs: list[Document], session_id: str):
    """Embeds chunks (dense + sparse) using cache and saves them to the Main Collection."""
    ensure_collection_exists(MAIN_COLLECTION)
    texts = [doc.page_content for doc in docs]

    dense_embeddings, sparse_embeddings = get_cached_embeddings(texts)
    
    points = []
    for i,doc in enumerate(docs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": sparse_embeddings[i]
                },
                payload={
                    "session_id":session_id,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            )
        )
    qdrant_client.upsert(
        collection_name=MAIN_COLLECTION,
        points=points
    )

def hybrid_search(query: str, session_id: str, k: int = 4, metadata_filter: dict = None) -> list[Document]:
    """Prefetch from both dense and sparse, fuse with Reciprocal Rank Fusion, applying filters."""
    if not qdrant_client.collection_exists(MAIN_COLLECTION):
        return []
    
    dense_query = list(dense_model.embed([query]))[0].tolist()
    sparse_query = list(sparse_model.embed([query]))[0]

    must_conditions = [FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    if metadata_filter:
        for key, value in metadata_filter.items():
            if value:  # Ignore empty filters
                must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))

    results = qdrant_client.query_points(
        collection_name=MAIN_COLLECTION,
        prefetch=[
            qdrant_models.Prefetch(
                query=dense_query,
                using="dense",
                limit=20,
            ),
            qdrant_models.Prefetch(
                query=qdrant_models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist()
                ),
                using="sparse",
                limit=20,
            ),
        ],
        query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
        query_filter=Filter(must=must_conditions),
        limit=k,
    )

    docs = []
    for hit in results.points:
        payload = hit.payload or {}
        docs.append(Document(
            page_content=payload.get("page_content", ""),
            metadata=payload.get("metadata", {})
        ))
    return docs

def check_cache(query: str , session_id: str) -> Optional[str]:
    """Checks if a semantically identical question was already answered."""
    ensure_collection_exists(CACHE_COLLECTION)
    
    query_vector = list(dense_model.embed([query]))[0].tolist()
    results = qdrant_client.query_points(
        collection_name=CACHE_COLLECTION,
        query=query_vector,
        limit=1,
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        )
    )
    
    if results.points and results.points[0].score >= CACHE_THRESHOLD:
        print(f"\n[CACHE] ⚡ Cache Hit! Score: {results.points[0].score}")
        return results.points[0].payload.get("answer")
    return None


def save_to_cache(query: str, answer: str, session_id: str):
    """Saves a new question and answer to the semantic cache."""
    ensure_collection_exists(CACHE_COLLECTION)
    
    query_vector = list(dense_model.embed([query]))[0].tolist()
    qdrant_client.upsert(
        collection_name=CACHE_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vector,
                payload={"session_id": session_id, "query": query, "answer": answer}
            )
        ]
    )