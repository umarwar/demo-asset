import logging
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from typing import List

from config.settings import Settings
from src.auth import authorize_request
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
    user_id: str = Form(...),
    _auth: None = Depends(authorize_request),
):
    """Upload a PDF file and trigger processing pipeline."""
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    file_size = len(file_bytes)
    max_size = Settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {Settings.MAX_FILE_SIZE_MB}MB",
        )

    # Validate user_id format
    try:
        UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    supabase = get_supabase_client()

    # Create document record in Supabase
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

    # Run processing in background
    background_tasks.add_task(process_document, file_bytes, file.filename, document_id, user_id)

    return UploadStatusResponse(
        document_id=document_id,
        filename=file.filename,
        status="uploading",
    )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(user_id: str, _auth: None = Depends(authorize_request)):
    """List all documents for a user."""
    try:
        UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

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
async def delete_document(document_id: str, user_id: str, _auth: None = Depends(authorize_request)):
    """Delete a document and remove its vectors from Pinecone."""
    try:
        UUID(document_id)
        UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    supabase = get_supabase_client()

    # Verify document exists and belongs to user
    result = (
        supabase.table("documents")
        .select("id, user_id, filename")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from Pinecone for this document
    try:
        namespace = f"user_{user_id}"
        pinecone_index = get_pinecone_index()

        # Delete by metadata filter
        pinecone_index.delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=namespace,
        )
    except Exception as e:
        logger.warning(f"Failed to delete vectors from Pinecone: {e}")

    # Delete from Supabase
    supabase.table("documents").delete().eq("id", document_id).execute()

    return {"message": "Document deleted", "document_id": document_id}


@router.get("/status/{document_id}", response_model=UploadStatusResponse)
async def get_upload_status(document_id: str, _auth: None = Depends(authorize_request)):
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
