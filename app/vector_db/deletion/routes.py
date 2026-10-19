from fastapi import APIRouter, HTTPException, status

from app.vector_db.deletion.deletion_utils import delete_by_filename, delete_namespace
from app.vector_db.models import DeleteResponseDto

router = APIRouter(prefix="/vector-db/delete", tags=["vector-db"])


def _raise_for_failure(result: DeleteResponseDto) -> DeleteResponseDto:
    if result.status:
        return result
    not_found = result.error is not None and "does not exist" in result.error
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if not_found else status.HTTP_502_BAD_GATEWAY,
        detail=result.error,
    )


@router.delete("/file/{index_name}/{namespace}/{filename}", response_model=DeleteResponseDto)
async def delete_file(index_name: str, namespace: str, filename: str) -> DeleteResponseDto:
    """Delete all chunks of one file from a namespace."""
    return _raise_for_failure(await delete_by_filename(index_name, namespace, filename))


@router.delete("/files/{index_name}/{namespace}", response_model=DeleteResponseDto)
async def delete_all_files(
    index_name: str, namespace: str, keep_namespace: bool = True
) -> DeleteResponseDto:
    """Delete all files in a namespace."""
    return _raise_for_failure(
        await delete_namespace(index_name, namespace, keep_namespace=keep_namespace)
    )
