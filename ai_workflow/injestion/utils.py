import asyncio

from langsmith import traceable
from loguru import logger

from ai_workflow.settings import config_settings
from ai_workflow.injestion.doc_loader import file_loader
from ai_workflow.injestion.models import RequestStatusEnum, IngestRequestDto, FileIngestionResponseDto
from ai_workflow.retrieval.answer_cache import invalidate_cached_answers
from ai_workflow.vector_db.utils import push_to_database, validate_and_create_index


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


if __name__ == "__main__":
    # Local test: python -m ai_workflow.injestion.utils
    from pathlib import Path

    from ai_workflow.vector_db.utils import close_vector_db

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
