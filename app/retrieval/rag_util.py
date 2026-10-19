from typing import Literal, Optional

from fastapi import WebSocket
from loguru import logger
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.settings import config_settings

MessageType = Literal["start", "stream", "end", "error", "info"]
ContentType = Literal["answer", "retrieval"]


class ChatResponse(BaseModel):
    """A server-to-client message."""

    message: str
    type: MessageType
    content_type: Optional[ContentType] = None
    code: Optional[str] = None


async def send_message_over_websocket(
    websocket: WebSocket,
    message: str,
    message_type: MessageType = "stream",
    content_type: Optional[ContentType] = None,
    code: Optional[str] = None,
) -> None:
    """Send one message; raises ``WebSocketDisconnect`` if the client is gone."""
    if websocket.client_state != WebSocketState.CONNECTED:
        raise WebSocketDisconnect(code=1006)
    frame = ChatResponse(
        message=message, type=message_type, content_type=content_type, code=code
    )
    try:
        await websocket.send_json(frame.model_dump())
    except RuntimeError as error:
        raise WebSocketDisconnect(code=1006) from error


class ConnectionManager:
    """Limits concurrent chat sockets."""

    def __init__(self, max_connections: Optional[int] = None) -> None:
        self.max_connections = max_connections or config_settings.MAX_WEBSOCKET_CONNECTIONS
        self._active: set[int] = set()

    @property
    def active_count(self) -> int:
        return len(self._active)

    def try_connect(self, websocket: WebSocket) -> bool:
        """Register the socket; False if at capacity."""
        if len(self._active) >= self.max_connections:
            logger.warning(
                f"WebSocket rejected: at capacity ({len(self._active)}/{self.max_connections})"
            )
            return False
        self._active.add(id(websocket))
        return True

    def disconnect(self, websocket: WebSocket) -> None:
        self._active.discard(id(websocket))
