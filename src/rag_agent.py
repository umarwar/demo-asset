import asyncio
import logging
from contextlib import suppress
from typing import List, Optional, AsyncGenerator

from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config.settings import Settings
from src.pinecone_client import get_pinecone_index

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intelligent document assistant powered by AMZU Consulting. You help users understand, analyze, and extract insights from their uploaded documents.

## YOUR ROLE:
Answer questions based on the user's uploaded documents. Be accurate, helpful, and friendly.

## AVAILABLE TOOLS & USAGE:
- `search_documents`: Search the user's uploaded documents for relevant information. Use this for ANY question about document content.

## RESPONSE RULES:
1. Answer simple greetings/confirmations directly, no tool call needed.
2. For ANY question that could possibly relate to document content — ALWAYS call `search_documents` FIRST before responding. Do NOT ask the user to upload documents without searching first. Search first, then respond based on results.
3. When answering from documents, cite the source filename (e.g., "According to **report.pdf**...").
4. If the tool returns no results, tell the user clearly: "I searched your documents but couldn't find information about that." NEVER pretend you found something when the tool returned nothing.
5. If the user claims they uploaded documents but the tool finds nothing, trust the tool results — do not agree with the user or pretend otherwise.
6. Politely decline clearly non-document topics (e.g., weather, coding help) and redirect to document-related questions.
7. Format responses using: bullet points for lists, bold for key details, and short paragraphs for explanations."""


class RAGAgent:
    """RAG Agent that retrieves chunks from Pinecone and answers questions."""

    def __init__(self):
        self.llm = OpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.LLM_TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
        )

    def _get_retriever(self, user_id: str):
        """Get a LlamaIndex retriever for a user's namespace."""
        namespace = f"user_{user_id}"
        pinecone_index = get_pinecone_index()

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=namespace,
        )

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        return index.as_retriever(similarity_top_k=Settings.SIMILARITY_TOP_K)

    def _create_agent(self, user_id: str) -> FunctionAgent:
        """Create a FunctionAgent with a document search tool for this user."""
        retriever = self._get_retriever(user_id)

        def search_documents(query: str) -> str:
            """Search the user's uploaded documents for relevant information.

            Args:
                query: The search query to find relevant document sections.

            Returns:
                Retrieved document chunks with source references.
            """
            nodes = retriever.retrieve(query)

            if not nodes:
                return "No relevant content found for this query in the documents. The documents may not cover this topic. Try rephrasing or asking about a different topic from the documents."

            results = []
            for i, node in enumerate(nodes, 1):
                filename = node.metadata.get("filename", "Unknown")
                score = f"{node.score:.3f}" if node.score else "N/A"
                results.append(
                    f"[Source {i}: {filename} (relevance: {score})]\n{node.text}"
                )

            return "\n\n---\n\n".join(results)

        search_tool = FunctionTool.from_defaults(fn=search_documents)

        return FunctionAgent(
            tools=[search_tool],
            llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
            streaming=True,
        )

    async def chat_streaming(
        self,
        message: str,
        user_id: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response with RAG retrieval."""
        if chat_history is None:
            chat_history = []

        agent = self._create_agent(user_id)

        try:
            context = Context(agent)
            handler = agent.run(message, ctx=context, chat_history=chat_history)
            stream = handler.stream_events()

            try:
                async for event in stream:
                    if hasattr(event, "delta") and event.delta:
                        yield event.delta
            finally:
                with suppress(Exception):
                    await stream.aclose()
                with suppress(asyncio.CancelledError):
                    await handler

        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"
