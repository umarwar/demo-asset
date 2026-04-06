import logging
import fitz  # pymupdf

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config.settings import Settings
from src.pinecone_client import get_pinecone_index
from src.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from PDF bytes using PyMuPDF.

    Returns:
        Tuple of (full_text, page_count)
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)
    pages_text = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            pages_text.append(text)

    doc.close()
    full_text = "\n\n".join(pages_text)
    return full_text, page_count


def chunk_and_index(
    text: str,
    filename: str,
    document_id: str,
    user_id: str,
) -> int:
    """
    Chunk text, embed, and index into Pinecone under user's namespace.

    Returns:
        Number of chunks created
    """
    namespace = f"user_{user_id}"

    # Create LlamaIndex Document with metadata
    document = Document(
        text=text,
        metadata={
            "filename": filename,
            "document_id": document_id,
        },
    )

    # Chunk using SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents([document])

    # Connect to Pinecone with user's namespace
    pinecone_index = get_pinecone_index()
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index nodes into Pinecone (embeds + upserts)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    return len(nodes)


def process_document(file_bytes: bytes, filename: str, document_id: str, user_id: str):
    """
    Full pipeline: extract text → chunk → embed → index into Pinecone.
    Updates document status in Supabase throughout.
    """
    supabase = get_supabase_client()

    try:
        # Update status to processing
        supabase.table("documents").update({"status": "processing"}).eq(
            "id", document_id
        ).execute()

        # Extract text
        text, page_count = extract_text_from_pdf(file_bytes)

        if not text.strip():
            supabase.table("documents").update(
                {"status": "failed", "page_count": page_count}
            ).eq("id", document_id).execute()
            logger.error(f"No text extracted from {filename}")
            return

        # Chunk and index
        chunk_count = chunk_and_index(text, filename, document_id, user_id)

        # Update status to ready
        supabase.table("documents").update(
            {
                "status": "ready",
                "page_count": page_count,
                "chunk_count": chunk_count,
            }
        ).eq("id", document_id).execute()

        logger.info(
            f"Document {filename} processed: {page_count} pages, {chunk_count} chunks"
        )

    except Exception as e:
        logger.error(f"Failed to process document {filename}: {e}")
        supabase.table("documents").update({"status": "failed"}).eq(
            "id", document_id
        ).execute()
