import re
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

PIPELINE_VERSION = "multimodal-v1"
ChunkType = Literal["text", "table", "image"]
# Fixed namespace: the same file content always yields the same chunk ids.
CHUNK_ID_NAMESPACE = uuid.UUID("2f7c1e0a-5b8d-4f61-9a3e-7d4c2b1a0e95")
FIGURE_LABEL_PATTERN = re.compile(r"^\s*(fig(?:ure)?\.?\s*\d+[a-z]?)", re.IGNORECASE)


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # common
    file_name: str
    file_type: str
    title: str
    file_hash: str
    chunk_id: str
    chunk_type: ChunkType
    chunk_index: int
    page: Optional[int] = None  # 1-based; None for DOCX
    total_pages: Optional[int] = None
    section: Optional[str] = None
    section_path: Optional[str] = None
    bbox: Optional[list[float]] = None
    part: int = 1
    parts: int = 1
    extractor: str
    pipeline_version: str = PIPELINE_VERSION
    ingested_at: str

    # text
    heading_level: Optional[int] = None

    # table
    table_index: Optional[int] = None
    table_title: Optional[str] = None
    columns: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None

    # image
    image_index: Optional[int] = None
    image_source: Optional[str] = None
    content_type: Optional[str] = None
    caption: Optional[str] = None
    figure_label: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_hash: Optional[str] = None
    vision_model: Optional[str] = None
    prompt_version: Optional[str] = None
    embedding_input: Optional[str] = None

    def to_metadata(self) -> dict:
        return self.model_dump(exclude_none=True, exclude_defaults=False)


def make_chunk_id(file_hash: str, order: int, part: int) -> str:
    """Deterministic UUID, also used as the Weaviate object id (re-runs overwrite, never duplicate)."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{file_hash}:{order}:{part}"))


def figure_label(caption: str) -> Optional[str]:
    """"Figure 3: ..." -> "Figure 3"."""
    match = FIGURE_LABEL_PATTERN.match(caption)
    return match.group(1) if match else None
