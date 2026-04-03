import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from llama_index.core.llms import ChatMessage, MessageRole
from supabase import Client

from src.supabase_client import get_supabase_client
from config.settings import Settings

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat history persistence in Supabase — exact GuidersAI pattern."""

    def __init__(self):
        self.client: Client = get_supabase_client()

    def get_or_create_chat(
        self,
        user_id: str,
        chat_id: Optional[str] = None,
        first_message: Optional[str] = None,
    ) -> str:
        """Get existing chat or create a new one. Returns chat_id."""
        try:
            UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user_id format: {user_id}")

        if chat_id:
            try:
                UUID(chat_id)
                result = (
                    self.client.table("chat_by_user")
                    .select("chat_id")
                    .eq("chat_id", chat_id)
                    .execute()
                )
                if result.data:
                    return chat_id
                else:
                    raise ValueError(f"Chat {chat_id} not found")
            except ValueError:
                raise ValueError(f"Invalid chat_id format: {chat_id}")

        # Create new chat
        new_chat_id = uuid4()
        title = None
        if first_message:
            title = first_message[:100].strip()
            if len(first_message) > 100:
                title += "..."

        result = (
            self.client.table("chat_by_user")
            .insert(
                {
                    "chat_id": str(new_chat_id),
                    "user_id": user_id,
                    "title": title,
                }
            )
            .execute()
        )

        if not result.data:
            raise RuntimeError("Failed to create new chat")

        return str(new_chat_id)

    def get_chat_history(self, chat_id: str, limit: int = 14) -> List[ChatMessage]:
        """Fetch chat history and convert to ChatMessage objects."""
        try:
            UUID(chat_id)
        except ValueError:
            raise ValueError(f"Invalid chat_id format: {chat_id}")

        result = (
            self.client.table("history_by_chat")
            .select("role, content, created")
            .eq("chat_id", chat_id)
            .order("created", desc=True)
            .limit(limit)
            .execute()
        )

        messages = []
        for row in result.data:
            role_str = row.get("role", "").lower()
            content = row.get("content", "")

            if role_str == "user":
                role = MessageRole.USER
            elif role_str == "assistant":
                role = MessageRole.ASSISTANT
            elif role_str == "system":
                role = MessageRole.SYSTEM
            else:
                role = MessageRole.USER

            messages.append(ChatMessage(role=role, content=content))

        messages.reverse()
        return messages

    def save_conversation(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
        user_message_id: str,
        assistant_message_id: str,
        created_user: int,
        created_assistant: int,
    ) -> None:
        """Save both user and assistant messages with specified IDs and timestamps."""
        try:
            UUID(chat_id)
            UUID(user_message_id)
            UUID(assistant_message_id)
        except ValueError as e:
            raise ValueError(f"Invalid UUID format: {e}")

        created_user_dt = datetime.fromtimestamp(created_user / 1000, tz=timezone.utc)
        created_assistant_dt = datetime.fromtimestamp(
            created_assistant / 1000, tz=timezone.utc
        )

        # Save user message
        user_result = (
            self.client.table("history_by_chat")
            .insert(
                {
                    "chat_id": chat_id,
                    "history_id": user_message_id,
                    "role": "user",
                    "content": user_message,
                    "created": created_user_dt.isoformat(),
                }
            )
            .execute()
        )

        if not user_result.data:
            raise RuntimeError(f"Failed to save user message to chat {chat_id}")

        # Save assistant message
        assistant_result = (
            self.client.table("history_by_chat")
            .insert(
                {
                    "chat_id": chat_id,
                    "history_id": assistant_message_id,
                    "role": "assistant",
                    "content": assistant_message,
                    "created": created_assistant_dt.isoformat(),
                }
            )
            .execute()
        )

        if not assistant_result.data:
            raise RuntimeError(
                f"Failed to save assistant message to chat {chat_id}"
            )

    def get_all_chats(self, user_id: str) -> List[dict]:
        """Fetch all chats for a user."""
        try:
            UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user_id format: {user_id}")

        result = (
            self.client.table("chat_by_user")
            .select("chat_id, user_id, created, title")
            .eq("user_id", user_id)
            .order("created", desc=True)
            .execute()
        )

        return result.data if result.data else []

    def get_all_messages(self, chat_id: str) -> List[dict]:
        """Fetch all messages for a chat."""
        try:
            UUID(chat_id)
        except ValueError:
            raise ValueError(f"Invalid chat_id format: {chat_id}")

        result = (
            self.client.table("history_by_chat")
            .select("chat_id, history_id, role, content, created")
            .eq("chat_id", chat_id)
            .order("created", desc=False)
            .execute()
        )

        return result.data if result.data else []
