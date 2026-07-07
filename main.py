from fastapi import FastAPI, HTTPException
from schemas import ChatRequest, ChatResponse, LoadDocumentRequest, LoadDocumentResponse
from services.paper_loader import load_document
from services.vector_store import add_paper
from services.rag_agent import run_query
 
app = FastAPI(
    title="Research Paper RAG API",
    description="An agentic backend for retrieving and analyzing scientific papers.",
    version="1.0.0"
)

@app.get("/health",tags=['System'])
def health_check():
    """
    Returns 200 OK if the API is running.
    """
    return {"status": "healthy", "message": "API is running smoothly."}

@app.post("/api/v1/documents/load", response_model=LoadDocumentResponse, tags=["Documents"])
def load_document_endpoint(req: LoadDocumentRequest):
    """
    Auto-detects the source type (ArXiv, Webpage, PDF, TXT),
    chunks it, saves Parents to SQLite, and embeds Children into Qdrant.
    """
    try:
        # 1. Use the auto-dispatcher!
        child_docs = load_document(req.source, req.session_id)
        
        # 2. Embed into Qdrant Main Collection
        add_paper(child_docs, req.session_id)
        
        return LoadDocumentResponse(
            status="success",
            message=f"Successfully loaded {req.source} and embedded {len(child_docs)} chunks."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(req: ChatRequest):
    """Sends a message to the Agentic RAG brain."""
    try:
        answer = run_query(req.query, req.session_id)
        
        return ChatResponse(
            answer=answer,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))