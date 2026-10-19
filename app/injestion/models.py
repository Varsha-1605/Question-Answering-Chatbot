from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


FILE_TYPE = [
    "pdf",
    "txt",
    "docx",
    "csv",
    "xlsx"
]


class RequestStatusEnum(Enum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FileIngestionResponseDto(BaseModel):
    file_path: str = None
    file_name: str = None
    error_detail: Optional[str] = None
    status: RequestStatusEnum = RequestStatusEnum.PROCESSING
    total_pages: Optional[int] = None
    # Multimodal ingestion only (None otherwise).
    images_described: Optional[int] = None
    images_failed: Optional[int] = None
    images_skipped: Optional[int] = None


class IngestRequestDto(BaseModel):
    pre_signed_url: str
    file_name: str
    original_file_name: str
    file_type: str
    index_name: Optional[str] = None
    namespace: Optional[str] = None
    delete_status: Optional[bool] = Field(
        default=False, alias="delete_status", description="Delete existing chunks of this file before storing"
    )
