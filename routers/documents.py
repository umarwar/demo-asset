import logging
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from typing import List

from config.settings import Settings
from src.auth import get_current_user
from src.supabase_client import get_supabase_client
from src.pinecone_client import get_pinecone_index
from src.document_processor import process_document
from src.models import DocumentResponse, UploadStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/upload", response_model=UploadStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload a PDF file and trigger processing pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()

    file_size = len(file_bytes)
    max_size = Settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {Settings.MAX_FILE_SIZE_MB}MB",
        )

    supabase = get_supabase_client()

    result = (
        supabase.table("documents")
        .insert(
            {
                "user_id": user_id,
                "filename": file.filename,
                "status": "uploading",
                "file_size_bytes": file_size,
            }
        )
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create document record")

    document_id = result.data[0]["id"]

    background_tasks.add_task(process_document, file_bytes, file.filename, document_id, user_id)

    return UploadStatusResponse(
        document_id=document_id,
        filename=file.filename,
        status="uploading",
    )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(user_id: str = Depends(get_current_user)):
    """List all documents for the authenticated user."""
    supabase = get_supabase_client()
    result = (
        supabase.table("documents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return result.data if result.data else []


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, user_id: str = Depends(get_current_user)):
    """Delete a document and remove its vectors from Pinecone."""
    try:
        UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    supabase = get_supabase_client()

    result = (
        supabase.table("documents")
        .select("id, user_id, filename")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        namespace = f"user_{user_id}"
        pinecone_index = get_pinecone_index()
        pinecone_index.delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=namespace,
        )
    except Exception as e:
        logger.warning(f"Failed to delete vectors from Pinecone: {e}")

    supabase.table("documents").delete().eq("id", document_id).execute()

    return {"message": "Document deleted", "document_id": document_id}


@router.get("/status/{document_id}", response_model=UploadStatusResponse)
async def get_upload_status(document_id: str, _user: str = Depends(get_current_user)):
    """Check processing status of an uploaded document."""
    try:
        UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    supabase = get_supabase_client()
    result = (
        supabase.table("documents")
        .select("id, filename, status, page_count, chunk_count")
        .eq("id", document_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = result.data[0]
    return UploadStatusResponse(
        document_id=doc["id"],
        filename=doc["filename"],
        status=doc["status"],
        page_count=doc.get("page_count"),
        chunk_count=doc.get("chunk_count"),
    )
