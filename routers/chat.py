import json
import logging
from datetime import datetime, timezone
from uuid import uuid4, UUID
from typing import List

from fastapi import APIRouter, HTTPException, Request, Depends
from sse_starlette.sse import EventSourceResponse

from config.settings import Settings
from src.auth import authorize_request
from src.models import (
    ChatRequest,
    ChatListRequest,
    ChatMessagesRequest,
    ChatListResponse,
    ChatMessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat_stream(request: ChatRequest, req: Request, _auth: None = Depends(authorize_request)):
    """Chat with documents via SSE streaming — same pattern as GuidersAI."""
    chat_history_mgr = req.app.state.chat_history
    agent = req.app.state.agent

    async def event_generator():
        try:
            # Get or create chat
            chat_id = chat_history_mgr.get_or_create_chat(
                user_id=request.user_id,
                chat_id=request.chat_id,
                first_message=request.message if request.chat_id is None else None,
            )

            # Load chat history
            chat_history = chat_history_mgr.get_chat_history(
                chat_id=chat_id, limit=Settings.CHAT_HISTORY_LIMIT
            )

            user_message_id = str(uuid4())
            assistant_message_id = str(uuid4())
            created_user = int(datetime.now(timezone.utc).timestamp() * 1000)
            created_assistant = created_user + 1

            assistant_response_parts = []

            try:
                async for chunk in agent.chat_streaming(
                    request.message,
                    user_id=request.user_id,
                    chat_history=chat_history,
                ):
                    assistant_response_parts.append(chunk)
                    yield {
                        "event": "token",
                        "data": chunk,
                    }

                # Save conversation after streaming completes
                assistant_response = "".join(assistant_response_parts)
                if assistant_response.strip():
                    try:
                        chat_history_mgr.save_conversation(
                            chat_id=chat_id,
                            user_message=request.message,
                            assistant_message=assistant_response,
                            user_message_id=user_message_id,
                            assistant_message_id=assistant_message_id,
                            created_user=created_user,
                            created_assistant=created_assistant,
                        )

                        metadata_payload = {
                            "chat_id": str(chat_id),
                            "history_id": str(assistant_message_id),
                            "created": created_assistant,
                        }
                        yield {
                            "event": "metadata",
                            "data": json.dumps(metadata_payload),
                        }
                        yield {
                            "event": "done",
                            "data": "true",
                        }

                    except Exception as save_error:
                        logger.error(f"Failed to save conversation: {save_error}")
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {"error": f"Failed to save conversation: {str(save_error)}"}
                            ),
                        }

            except Exception as exc:
                logger.error(f"Error during streaming: {exc}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(exc)}),
                }

        except ValueError as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
        except Exception as e:
            logger.error(f"Unexpected error in chat_stream: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Internal error: {str(e)}"}),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/all", response_model=List[ChatListResponse])
async def get_all_chats(request: ChatListRequest, req: Request, _auth: None = Depends(authorize_request)):
    """Get all chats for a user."""
    chat_history_mgr = req.app.state.chat_history

    try:
        chats = chat_history_mgr.get_all_chats(request.user_id)

        response_list = []
        for chat in chats:
            created = chat.get("created")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created = datetime.now(timezone.utc)
            elif created is None:
                created = datetime.now(timezone.utc)

            response_list.append(
                ChatListResponse(
                    user_id=UUID(chat["user_id"]),
                    chat_id=UUID(chat["chat_id"]),
                    created=created,
                    title=chat.get("title") or "Untitled Chat",
                )
            )

        return response_list
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching chats: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/chat/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(request: ChatMessagesRequest, req: Request, _auth: None = Depends(authorize_request)):
    """Get all messages for a chat."""
    chat_history_mgr = req.app.state.chat_history

    try:
        messages = chat_history_mgr.get_all_messages(request.chat_id)

        response_list = []
        for msg in messages:
            created = msg.get("created")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created = datetime.now(timezone.utc)
            elif created is None:
                created = datetime.now(timezone.utc)

            response_list.append(
                ChatMessageResponse(
                    chat_id=UUID(msg["chat_id"]),
                    history_id=UUID(msg["history_id"]),
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    created=created,
                )
            )

        return response_list
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
