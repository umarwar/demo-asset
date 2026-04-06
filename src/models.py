from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


# ── Chat Models (exact GuidersAI schema) ──

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None


class ChatListResponse(BaseModel):
    user_id: UUID
    chat_id: UUID
    created: datetime
    title: str


class ChatMessageResponse(BaseModel):
    chat_id: UUID
    history_id: UUID
    role: str
    content: str
    created: datetime


# ── Document Models ──

class DocumentResponse(BaseModel):
    id: UUID
    user_id: UUID
    filename: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    created_at: datetime


class UploadStatusResponse(BaseModel):
    document_id: UUID
    filename: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
