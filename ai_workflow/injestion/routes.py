import asyncio
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from ai_workflow.injestion.models import (
    FILE_TYPE,
    FileIngestionResponseDto,
    IngestRequestDto,
    RequestStatusEnum,
)
from ai_workflow.injestion.utils import load_file_and_push_to_database
from ai_workflow.settings import config_settings

router = APIRouter(tags=['injestion'])

READ_BLOCK_BYTES = 1024 * 1024


async def ingest_one_file(
        upload: UploadFile,
        file_name: str,
        file_type: str,
) -> FileIngestionResponseDto:
    """Ingest an upload via a temp file; index and namespace come from settings, not the client."""
    max_bytes = config_settings.MAX_FILE_SIZE_MB * 1024 * 1024
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}")
    try:
        with temp:
            written = 0
            while block := await upload.read(READ_BLOCK_BYTES):
                written += len(block)
                if written > max_bytes:
                    return FileIngestionResponseDto(
                        file_name=file_name,
                        status=RequestStatusEnum.FAILED,
                        error_detail=f"File is larger than the {config_settings.MAX_FILE_SIZE_MB} MB limit",
                    )
                temp.write(block)

        return await load_file_and_push_to_database(
            request=IngestRequestDto(
                pre_signed_url=temp.name,
                file_name=file_name,
                original_file_name=file_name,
                file_type=file_type,
                index_name=config_settings.INDEX_NAME,
                namespace=config_settings.DEFAULT_NAMESPACE,
                delete_status=False,
            )
        )
    finally:
        os.remove(temp.name)


@router.post(
    "/ingest",
    response_model=List[FileIngestionResponseDto],
    status_code=status.HTTP_200_OK,
)
async def ingest_doc(
        # format=binary makes Swagger UI show a file chooser
        files: List[UploadFile] = File(
            ..., json_schema_extra={"items": {"type": "string", "format": "binary"}}
        ),
) -> List[FileIngestionResponseDto]:
    """Store uploaded files in the vector database."""
    uploads = []
    for upload in files:
        file_name = Path(upload.filename or "").name
        file_type = Path(file_name).suffix.lstrip(".").lower()
        if file_type not in FILE_TYPE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_type or file_name}. Supported: {', '.join(FILE_TYPE)}"
            )
        uploads.append((upload, file_name, file_type))

    results = await asyncio.gather(
        *(
            ingest_one_file(upload, name, kind)
            for upload, name, kind in uploads
        )
    )

    logger.info(f"Successfully processed {len(results)} files.")
    return list(results)
