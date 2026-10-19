from typing import Literal

from pydantic import BaseModel, Field

from app.settings import config_settings


class Message(BaseModel):
    type: Literal["human", "ai"]
    content: str = Field(max_length=20_000)


class TitleRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config_settings.MAX_QUESTION_CHARS)


class ChatRequest(BaseModel):
    """A user question sent over the WebSocket."""

    question: str = Field(min_length=1, max_length=config_settings.MAX_QUESTION_CHARS)
    language: str = Field(default="English", min_length=1, max_length=40)
    chat_context: list[Message] = Field(
        default_factory=list, max_length=config_settings.MAX_CHAT_CONTEXT_MESSAGES
    )
