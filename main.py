import logging
import time
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from config.settings import Settings
from src.embedding import configure_embedding
from src.supabase_client import get_supabase_client
from src.chat_history import ChatHistoryManager
from src.rag_agent import RAGAgent
from routers.documents import router as documents_router
from routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AMZU RAG Demo Asset")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    start_time = time.time()

    # Configure LlamaIndex embedding model globally
    configure_embedding()

    # Validate Supabase credentials
    if not Settings.SUPABASE_URL or not Settings.SUPABASE_KEY:
        logger.warning("Supabase credentials not set. Chat history will not persist.")

    # Warm Supabase connection
    get_supabase_client()

    # Initialize chat history manager (once, reuse across all requests)
    app.state.chat_history = ChatHistoryManager()

    # Initialize RAG agent (once, reuse across all requests)
    app.state.agent = RAGAgent()

    elapsed = time.time() - start_time
    logger.info(f"Server ready ({elapsed:.2f}s)")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Register routers
app.include_router(documents_router)
app.include_router(chat_router)
