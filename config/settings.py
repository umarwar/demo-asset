import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class Settings:
    """Central configuration for RAG Demo Asset"""

    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")

    # Supabase
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.1

    # RAG Settings
    SIMILARITY_TOP_K = 5
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200

    # Chat Settings
    CHAT_HISTORY_LIMIT = 14

    # Access API Key
    ACCESS_API_KEY = os.environ.get("ACCESS_API_KEY")

    # Upload
    MAX_FILE_SIZE_MB = 50
    ALLOWED_EXTENSIONS = {".pdf"}
