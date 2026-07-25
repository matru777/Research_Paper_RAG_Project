import uuid
from datetime import datetime
from typing import List, Optional, Any
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    sessions: Mapped[List["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String, default="New Research Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    user: Mapped["User"] = relationship(back_populates="sessions")
    jobs: Mapped[List["ProcessingJob"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    documents: Mapped[List["ParentDocument"]] = relationship(back_populates="session", cascade="all, delete-orphan")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    file_name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="PENDING") 
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    session: Mapped["Session"] = relationship(back_populates="jobs")

class ParentDocument(Base):
    __tablename__ = "parent_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    content: Mapped[str] = mapped_column(String)
    document_metadata: Mapped[dict[str,Any]] = mapped_column(JSONB, default=dict)
    session: Mapped["Session"] = relationship(back_populates="documents")


    