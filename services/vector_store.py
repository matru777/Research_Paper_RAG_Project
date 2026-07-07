import uuid
from typing import Optional
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue ,PayloadSchemaType
from core.config import Settings
from fastembed import TextEmbedding

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant_client = QdrantClient(
    url=Settings.QDRANT_URL,
    api_key=Settings.QDRANT_API_KEY,
    timeout=60
)

MAIN_COLLECTION = "research_papers"
CACHE_COLLECTION = "rpaper_semantic_cache_v3"
CACHE_THRESHOLD = 0.90 

def ensure_collection_exists(collection_name: str, vector_size: int = 384):
    """Creates the Qdrant collection if it doesn't exist."""
    if not qdrant_client.collection_exists(collection_name):
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
    """Embeds chunks and saves them to the Main Collection tagged with the session_id."""
    ensure_collection_exists(MAIN_COLLECTION)
    texts = [doc.page_content for doc in docs]
    embeddings = list(embedding_model.embed(texts))
    points = []
    for i,doc in enumerate(docs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
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

def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Searches the Main Collection, filtering strictly by session_id."""
    if not qdrant_client.collection_exists(MAIN_COLLECTION):
        return []
    
    query_vector = list(embedding_model.embed([query]))[0].tolist()

    results = qdrant_client.query_points(
        collection_name=MAIN_COLLECTION,
        query=query_vector,
        limit=k,
        # <--- MULTI-TENANCY FILTER
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        )
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
    
    query_vector = list(embedding_model.embed([query]))[0].tolist()
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
    
    query_vector = list(embedding_model.embed([query]))[0].tolist()
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
