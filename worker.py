import asyncio
from arq.connections import RedisSettings
from sqlalchemy import select

from core.database import AsyncSessionLocal
from core.models import ProcessingJob
from services.paper_loader import load_document
from services.vector_store import add_paper

async def process_document_task(ctx, job_id:str, source:str, session_id:str):

    async with AsyncSessionLocal() as session:
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        job = (await session.execute(stmt)).scalar_one_or_none()
        if job:
            job.status = "PROCESSING"
            await session.commit()

    try:
        child_docs = await load_document(source, session_id)
        add_paper(child_docs, session_id)

        async with AsyncSessionLocal() as session:
            stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
            job = (await session.execute(stmt)).scalar_one_or_none()
            if job:
                job.status = "COMPLETED"
                job.progress = 100
                await session.commit()

    except Exception as e:
        async with AsyncSessionLocal() as session:
            stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
            job = (await session.execute(stmt)).scalar_one_or_none()
            if job:
                job.status = "FAILED"
                job.error_message = str(e)
                await session.commit()
        raise e

class WorkerSettings:
    functions = [process_document_task]
    redis_settings = RedisSettings(host="localhost", port=6380)
    job_timeout = 600  # Allow 10 minutes for Docling
    max_tries = 1      # Don't infinitely retry failed jobs