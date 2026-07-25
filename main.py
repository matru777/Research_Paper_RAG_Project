from fastapi import FastAPI, HTTPException
from schemas import ChatRequest, ChatResponse, LoadDocumentRequest, LoadDocumentResponse
from services.paper_loader import load_document
from services.vector_store import add_paper
from services.rag_pipeline import run_pipeline_stream
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import json
from arq.connections import RedisSettings

from arq import create_pool
from core.database import AsyncSessionLocal
from core.models import ProcessingJob, Session, User
from sqlalchemy import select
import uuid

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

@app.post("/api/v1/documents/load",tags=["Documents"])
async def load_document_endpoint(req: LoadDocumentRequest):
    """
    Asynchronous Ingestion: Writes the job to Postgres and pushes it to Redis.
    """
    job_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        # ⚡ Auto-create Guest User & Session if missing to satisfy Foreign Keys ⚡
        stmt = select(Session).where(Session.id == req.session_id)
        existing_session = (await session.execute(stmt)).scalar_one_or_none()
        
        if not existing_session:
            user_stmt = select(User).where(User.email == "guest@local")
            guest_user = (await session.execute(user_stmt)).scalar_one_or_none()
            if not guest_user:
                guest_user = User(email="guest@local")
                session.add(guest_user)
                await session.flush()
                
            new_session = Session(id=req.session_id, user_id=guest_user.id)
            session.add(new_session)
            await session.flush()

        new_job = ProcessingJob(
            id=job_id,  
            session_id=req.session_id,
            file_name=req.source,
            status="PENDING"
        )

        session.add(new_job)
        await session.commit()

    redis = await create_pool(RedisSettings(host="localhost", port=6380))

    await redis.enqueue_job(
        "process_document_task",
        str(job_id),
        req.source,
        req.session_id
    )

    return {
        "status": "success",
        "job_id": str(job_id),
        "message": f"Job submitted to background worker."
    }

@app.get("/api/v1/documents/status/{job_id}", tags=["Documents"])
async def get_job_status(job_id: str):
    """
    Allows the Frontend UI to poll the status of a PDF upload.
    """
    async with AsyncSessionLocal() as session:
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": str(job.id),
            "file_name": job.file_name,
            "status": job.status,
            "progress": job.progress,
            "error_message": job.error_message
        }

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(req: ChatRequest):

    """Sends a message to the RAG Pipeline synchronously."""
    try:
        answer = ""
        async for chunk in run_pipeline_stream(req.query, req.session_id):
            answer += chunk
            
        return ChatResponse(
            answer=answer,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(req: ChatRequest):
    """Streams the AI response token-by-token via SSE."""
    async def event_generator():
        try:
            async for token in run_pipeline_stream(req.query, req.session_id):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")