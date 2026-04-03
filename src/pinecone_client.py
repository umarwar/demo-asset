import logging
from functools import lru_cache
from pinecone import Pinecone
from config.settings import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    """Create (or reuse) a Pinecone client."""
    if not Settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required")

    try:
        pc = Pinecone(api_key=Settings.PINECONE_API_KEY)
        logger.info("Pinecone client initialized")
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
        raise


def get_pinecone_index():
    """Get the Pinecone index instance connected via host."""
    if not Settings.PINECONE_INDEX_HOST:
        raise ValueError("PINECONE_INDEX_HOST is required")

    pc = get_pinecone_client()
    return pc.Index(host=Settings.PINECONE_INDEX_HOST)
