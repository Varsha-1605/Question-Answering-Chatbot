import asyncio
import os

from langsmith import traceable
from loguru import logger

from app.settings import config_settings
from app.injestion.doc_loader import file_loader, hash_file, resolve_local_file
from app.injestion.models import RequestStatusEnum, IngestRequestDto, FileIngestionResponseDto
from app.injestion.multimodal import ingestion_fingerprint, is_multimodal, load_multimodal_document
from app.retrieval.answer_cache import invalidate_cached_answers
from app.utils import get_image_embedder
from app.vector_db.utils import (
    is_file_unchanged,
    push_documents_with_images,
    push_to_database,
    validate_and_create_index,
)


@traceable(
    name="ingest_file",
    run_type="chain",
    # pre_signed_url may contain a credential, so it is not traced.
    process_inputs=lambda inputs: {
        "file_name": inputs["request"].file_name,
        "file_type": inputs["request"].file_type,
        "index_name": inputs["request"].index_name,
        "namespace": inputs["request"].namespace,
        "delete_status": inputs["request"].delete_status,
    },
    process_outputs=lambda response: {
        "status": response.status.value,
        "chunks": response.total_pages,
        "error": response.error_detail,
    },
)
async def load_file_and_push_to_database(
        request: IngestRequestDto,
) -> FileIngestionResponseDto:
    """Load a file and store its chunks in the vector database."""
    logger.info(
        f"Received request for injestion: {request.model_dump_json(exclude_none=True, exclude_unset=True, exclude_defaults=True)}"
    )

    if not request.namespace:
        request.namespace = config_settings.DEFAULT_NAMESPACE

    if not request.index_name:
        request.index_name = config_settings.INDEX_NAME

    try:
        await validate_and_create_index(index_name=request.index_name)

        if is_multimodal(request.file_type):
            return await ingest_multimodal_file(request)

        chunked_documents = await asyncio.to_thread(
            file_loader,
            pre_signed_url=request.pre_signed_url,
            file_name=request.file_name,
            file_type=request.file_type,
        )

        logger.info(f"Loaded {len(chunked_documents)} documents from {request.file_name}")

        if not chunked_documents:
            return FileIngestionResponseDto(
                file_path=request.pre_signed_url,
                file_name=request.file_name,
                status=RequestStatusEnum.FAILED,
                error_detail="No documents extracted from file"
            )

        push_response = await push_to_database(
            texts=chunked_documents,
            index_name=request.index_name,
            namespace=request.namespace,
            drop_status=request.delete_status,
        )

        if push_response.status:
            if push_response.chunks_stored:
                await invalidate_cached_answers(request.index_name, request.namespace)
            return FileIngestionResponseDto(
                file_path=request.pre_signed_url,
                file_name=request.file_name,
                status=RequestStatusEnum.COMPLETED,
                total_pages=len(chunked_documents)
            )

        logger.error(f"Failed to upload documents: {push_response.error}")
        return FileIngestionResponseDto(
            file_path=request.pre_signed_url,
            file_name=request.file_name,
            status=RequestStatusEnum.FAILED,
            error_detail=push_response.error
        )

    except Exception as e:
        logger.error(f"Error processing file {request.file_name}: {str(e)}")
        return FileIngestionResponseDto(
            file_path=request.pre_signed_url,
            file_name=request.file_name,
            status=RequestStatusEnum.FAILED,
            error_detail=str(e)
        )


async def ingest_multimodal_file(request: IngestRequestDto) -> FileIngestionResponseDto:
    """PDF/DOCX with MULTIMODAL_ENABLED: text, tables and images processed separately.

    An unchanged file (same bytes, same models and pipeline) is skipped before
    any parsing or vision call. A changed file replaces its previous version.
    """
    local_path, temp_path = await asyncio.to_thread(resolve_local_file, request.pre_signed_url, request.file_type)
    try:
        file_hash = await asyncio.to_thread(hash_file, local_path)
        fingerprint = ingestion_fingerprint(file_hash)
        if await is_file_unchanged(request.index_name, request.namespace, request.file_name, fingerprint):
            logger.info(f"{request.file_name} unchanged in {request.index_name}/{request.namespace}; skipping")
            return _response(request, RequestStatusEnum.COMPLETED, total_pages=0)

        loaded = await load_multimodal_document(
            local_path, request.file_name, request.file_type, file_hash,
            embeds_images=get_image_embedder() is not None,
        )
        image_counts = {
            "images_described": loaded.images_described,
            "images_failed": loaded.images_failed,
            "images_skipped": loaded.images_skipped,
        }
        if not loaded.chunks:
            return _response(request, RequestStatusEnum.FAILED, error_detail="No content extracted from file", **image_counts)

        push = await push_documents_with_images(
            documents=[chunk.document for chunk in loaded.chunks],
            images=[chunk.image.image_bytes if chunk.image else None for chunk in loaded.chunks],
            index_name=request.index_name,
            namespace=request.namespace,
            fingerprint=fingerprint,
        )
        if not push.status:
            return _response(request, RequestStatusEnum.FAILED, error_detail=push.error, **image_counts)

        await invalidate_cached_answers(request.index_name, request.namespace)
        # Stored, but some images could not be described: say so instead of a silent success.
        error_detail = f"{loaded.images_failed} image(s) could not be described" if loaded.images_failed else None
        return _response(
            request, RequestStatusEnum.COMPLETED, total_pages=len(loaded.chunks),
            error_detail=error_detail, **image_counts,
        )
    finally:
        if temp_path:
            await asyncio.to_thread(os.remove, temp_path)


def _response(request: IngestRequestDto, status: RequestStatusEnum, **fields) -> FileIngestionResponseDto:
    return FileIngestionResponseDto(
        file_path=request.pre_signed_url, file_name=request.file_name, status=status, **fields
    )


if __name__ == "__main__":
    # Local test: python -m app.injestion.utils
    from pathlib import Path

    from app.vector_db.utils import close_vector_db

    FILE_PATH = "data/transformer_paper.pdf"

    async def main():
        path = Path(FILE_PATH)
        try:
            result = await load_file_and_push_to_database(
                IngestRequestDto(
                    pre_signed_url=str(path),
                    file_name=path.name,
                    original_file_name=path.name,
                    file_type=path.suffix.lstrip(".").lower(),
                    delete_status=True,
                )
            )
        finally:
            close_vector_db()

        print(f"\nIndex/namespace: {config_settings.INDEX_NAME}/{config_settings.DEFAULT_NAMESPACE}")
        print(f"File:            {result.file_name}")
        print(f"Status:          {result.status.value}")
        print(f"Chunks:          {result.total_pages}")
        if result.error_detail:
            print(f"Error:           {result.error_detail}")

    asyncio.run(main())
