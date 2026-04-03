import logging
from functools import lru_cache
from supabase import create_client, Client
from config.settings import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create (or reuse) a Supabase client. Cached via lru_cache."""
    if not Settings.SUPABASE_URL or not Settings.SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")

    try:
        client = create_client(Settings.SUPABASE_URL, Settings.SUPABASE_KEY)
        logger.info("Supabase client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise
