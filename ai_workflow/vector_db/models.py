from typing import Optional

from pydantic import BaseModel


class PushToDatabaseResponseDto(BaseModel):
    status: bool = False
    chunks_stored: int = 0
    index: Optional[str] = None
    namespace: Optional[str] = None
    error: Optional[str] = None


class DeleteResponseDto(BaseModel):
    status: bool = False
    index: Optional[str] = None
    namespace: Optional[str] = None
    filename: Optional[str] = None
    deleted_count: Optional[int] = None
    error: Optional[str] = None
