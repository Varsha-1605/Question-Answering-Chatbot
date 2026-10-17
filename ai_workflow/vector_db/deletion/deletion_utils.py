import asyncio

from loguru import logger

from ai_workflow.retrieval.answer_cache import invalidate_cached_answers
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.models import DeleteResponseDto
from ai_workflow.vector_db.weaviate_utils.weaviate_client import db_connection


def _resolve(index_name: str | None, namespace: str | None) -> tuple[str, str]:
    return (
        index_name or config_settings.INDEX_NAME,
        namespace or config_settings.DEFAULT_NAMESPACE,
    )


async def delete_by_filename(
    index_name: str | None, namespace: str | None, filename: str
) -> DeleteResponseDto:
    """Delete all chunks of a file in a namespace."""
    index_name, namespace = _resolve(index_name, namespace)
    try:
        if not await asyncio.to_thread(db_connection.tenant_exists, index_name, namespace):
            return DeleteResponseDto(
                status=False,
                index=index_name,
                namespace=namespace,
                filename=filename,
                error=f"Namespace '{namespace}' does not exist in index '{index_name}'",
            )
        deleted = await asyncio.to_thread(
            db_connection.delete_by_file_name, index_name, namespace, filename
        )
        logger.info(f"Deleted {deleted} chunks of {filename} from {index_name}/{namespace}")
        if deleted:
            await invalidate_cached_answers(index_name, namespace)
        return DeleteResponseDto(
            status=True,
            index=index_name,
            namespace=namespace,
            filename=filename,
            deleted_count=deleted,
        )
    except Exception as error:  # noqa: BLE001
        logger.exception(f"Failed to delete {filename} from {index_name}/{namespace}")
        return DeleteResponseDto(
            status=False,
            index=index_name,
            namespace=namespace,
            filename=filename,
            error=str(error),
        )


async def delete_namespace(
    index_name: str | None, namespace: str | None, keep_namespace: bool = False
) -> DeleteResponseDto:
    """Delete a namespace's data, optionally keeping the empty namespace."""
    index_name, namespace = _resolve(index_name, namespace)
    try:
        if not await asyncio.to_thread(db_connection.tenant_exists, index_name, namespace):
            return DeleteResponseDto(
                status=False,
                index=index_name,
                namespace=namespace,
                error=f"Namespace '{namespace}' does not exist in index '{index_name}'",
            )
        operation = db_connection.clear_tenant if keep_namespace else db_connection.delete_tenant
        await asyncio.to_thread(operation, index_name, namespace)
        logger.info(f"Deleted namespace {index_name}/{namespace} (keep={keep_namespace})")
        await invalidate_cached_answers(index_name, namespace)
        return DeleteResponseDto(status=True, index=index_name, namespace=namespace)
    except Exception as error:  # noqa: BLE001
        logger.exception(f"Failed to delete namespace {index_name}/{namespace}")
        return DeleteResponseDto(
            status=False, index=index_name, namespace=namespace, error=str(error)
        )
