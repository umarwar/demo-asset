from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from config.settings import Settings


def configure_embedding():
    """Configure LlamaIndex global embedding model."""
    embed_model = OpenAIEmbedding(
        model=Settings.EMBEDDING_MODEL,
        dimensions=Settings.EMBEDDING_DIMENSION,
        api_key=Settings.OPENAI_API_KEY,
    )
    LlamaSettings.embed_model = embed_model
    return embed_model
