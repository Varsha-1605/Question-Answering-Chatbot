import asyncio
import hashlib

from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from langsmith import traceable
from loguru import logger

from ai_workflow.settings import config_settings
from ai_workflow.utils import get_embeddings
from ai_workflow.vector_db.models import PushToDatabaseResponseDto
from ai_workflow.vector_db.sanitization import sanitize_index_name, sanitize_namespace
from ai_workflow.vector_db.weaviate_utils.weaviate_client import (
    CONTENT_HASH_PROPERTY,
    FILE_NAME_PROPERTY,
    db_connection,
)

# One store per index; rebuilt when the client reconnects.
_stores: dict[str, tuple[object, WeaviateVectorStore]] = {}


def get_vector_store(index_name: str) -> WeaviateVectorStore:
    """Return the cached vector store for an index."""
    name = sanitize_index_name(index_name)
    client = db_connection.get_client()
    cached = _stores.get(name)
    if cached is not None and cached[0] is client:
        return cached[1]

    store = WeaviateVectorStore(
        client=client,
        index_name=name,
        text_key=config_settings.WEAVIATE_TEXT_KEY,
        embedding=get_embeddings(),
        use_multi_tenancy=True,
    )
    _stores[name] = (client, store)
    return store


def content_hash(documents: list[Document]) -> str:
    """Hash of all chunk texts, in order."""
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.page_content.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@traceable(
    name="store_in_weaviate",
    run_type="tool",
    process_inputs=lambda inputs: {
        "index_name": inputs["index_name"],
        "tenant": inputs["tenant"],
        "chunks": len(inputs["documents"]),
        "file_name": inputs["documents"][0].metadata.get(FILE_NAME_PROPERTY),
    },
)
def write_chunks(index_name: str, tenant: str, documents: list[Document]) -> None:
    """Embed and write chunks to the tenant (blocking)."""
    get_vector_store(index_name).add_documents(documents, tenant=tenant)


def store_documents(
    documents: list[Document],
    index_name: str,
    namespace: str,
    drop_status: bool = False,
) -> int:
    """Embed and store one file's chunks; returns chunks written (0 if unchanged).

    ``drop_status`` deletes the file's existing chunks first.
    """
    if not documents:
        return 0

    index = db_connection.ensure_collection(index_name)
    tenant = sanitize_namespace(namespace)
    file_name = documents[0].metadata.get(FILE_NAME_PROPERTY)
    fingerprint = content_hash(documents)

    if file_name and db_connection.tenant_exists(index, tenant):
        if db_connection.get_content_hash(index, tenant, file_name) == fingerprint:
            logger.info(f"{file_name} unchanged in {index}/{tenant}; skipping embedding")
            return 0
        if drop_status:
            removed = db_connection.delete_by_file_name(index, tenant, file_name)
            logger.info(f"Removed {removed} existing chunks of {file_name} in {index}/{tenant}")

    for doc in documents:
        doc.metadata[CONTENT_HASH_PROPERTY] = fingerprint

    write_chunks(index, tenant, documents)
    logger.info(f"Stored {len(documents)} chunks in {index}/{tenant}")
    return len(documents)


async def validate_and_create_index(index_name: str) -> bool:
    """Create the index if it doesn't exist."""
    await asyncio.to_thread(db_connection.ensure_collection, index_name)
    return True


async def push_to_database(
    texts: list[Document],
    index_name: str,
    namespace: str,
    drop_status: bool = False,
) -> PushToDatabaseResponseDto:
    """Embed and store chunks; failures are returned in the response."""
    try:
        stored = await asyncio.to_thread(
            store_documents, texts, index_name, namespace, drop_status
        )
        return PushToDatabaseResponseDto(
            status=True, chunks_stored=stored, index=index_name, namespace=namespace
        )
    except Exception as error:  # noqa: BLE001
        logger.exception(f"Failed to push documents to {index_name}/{namespace}")
        return PushToDatabaseResponseDto(
            status=False, index=index_name, namespace=namespace, error=str(error)
        )


async def has_documents(index_name: str, namespace: str) -> bool:
    """True if the namespace has at least one chunk."""
    return await asyncio.to_thread(db_connection.tenant_has_objects, index_name, namespace)


async def is_vector_db_ready() -> bool:
    return await asyncio.to_thread(db_connection.is_ready)


def close_vector_db() -> None:
    db_connection.close()
