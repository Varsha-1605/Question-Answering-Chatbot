import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from langsmith import traceable
from loguru import logger

from app.injestion.multimodal.chunking import Chunk, FileInfo, build_chunks
from app.injestion.multimodal.docx_extractor import extract_docx
from app.injestion.multimodal.elements import (
    ExtractedDocument,
    ExtractionLimits,
    ImageElement,
)
from app.injestion.multimodal.metadata import PIPELINE_VERSION
from app.injestion.multimodal.pdf_extractor import extract_pdf
from app.injestion.multimodal.prompts import PROMPT_VERSION
from app.injestion.multimodal.vision import get_image_describer, vision_model_id
from app.settings import config_settings
from app.utils import embedding_model_id

MULTIMODAL_FILE_TYPES = frozenset({"pdf", "docx"})
EXTRACTORS = {
    "pdf": (extract_pdf, "pymupdf"),
    "docx": (extract_docx, "python-docx"),
}


@dataclass(frozen=True, slots=True)
class MultimodalLoadResult:
    chunks: list[Chunk]
    images_described: int
    images_failed: int
    images_skipped: int  # tiny, repeated, decorative, unreadable or over the per-file cap


def is_multimodal(file_type: str) -> bool:
    return config_settings.MULTIMODAL_ENABLED and file_type.lower() in MULTIMODAL_FILE_TYPES


def ingestion_fingerprint(file_hash: str) -> str:
    """Version key stored with every chunk of a file.

    Covers the file bytes and everything that shapes its chunks (embedding model,
    vision model, prompt, pipeline), so changing any of them re-ingests the file
    instead of skipping it as unchanged.
    """
    parts = (file_hash, embedding_model_id(), vision_model_id(), PROMPT_VERSION, PIPELINE_VERSION)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


@traceable(
    name="load_multimodal_file",
    run_type="chain",
    # Only names and counts are traced: no paths, image bytes or document text.
    process_inputs=lambda inputs: {"file_name": inputs["file_name"], "file_type": inputs["file_type"]},
    process_outputs=lambda result: {
        "chunks": len(result.chunks),
        "images_described": result.images_described,
        "images_failed": result.images_failed,
        "images_skipped": result.images_skipped,
    },
)
async def load_multimodal_document(
    path: str, file_name: str, file_type: str, file_hash: str, embeds_images: bool
) -> MultimodalLoadResult:
    file_type = file_type.lower()
    extract, extractor_name = EXTRACTORS[file_type]
    limits = ExtractionLimits(
        min_image_px=config_settings.VISION_MIN_IMAGE_PX,
        max_images=config_settings.VISION_MAX_IMAGES_PER_FILE,
    )

    extracted: ExtractedDocument = await asyncio.to_thread(extract, path, limits)
    images = [element for element in extracted.elements if isinstance(element, ImageElement)]
    vision = await get_image_describer().describe(images)

    file = FileInfo(
        file_name=file_name,
        file_type=file_type,
        file_hash=file_hash,
        extractor=extractor_name,
        ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
        total_pages=extracted.total_pages,
    )
    chunks = build_chunks(
        extracted, vision, file,
        chunk_size=config_settings.CHUNK_SIZE,
        chunk_overlap=config_settings.CHUNK_OVERLAP,
        vision_model=vision_model_id(),
        embeds_images=embeds_images,
    )

    decorative = sum(1 for description in vision.descriptions.values() if description.is_decorative)
    described = len(vision.descriptions) - decorative
    logger.info(
        f"{file_name}: {len(extracted.elements)} elements, {len(images)} images "
        f"({described} described, {vision.failed} failed, {extracted.images_skipped + decorative} skipped), "
        f"{len(chunks)} chunks"
    )

    return MultimodalLoadResult(
        chunks=chunks,
        images_described=described,
        images_failed=vision.failed,
        images_skipped=extracted.images_skipped + decorative,
    )
