import asyncio
import hashlib
from typing import Optional

from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from langsmith import traceable
from loguru import logger

from app.exception import VectorStoreError
from app.settings import config_settings
from app.utils import get_embeddings, get_image_embedder
from app.vector_db.models import PushToDatabaseResponseDto
from app.vector_db.sanitization import sanitize_index_name, sanitize_namespace
from app.vector_db.weaviate_utils.weaviate_client import (
    CONTENT_HASH_PROPERTY,
    FILE_NAME_PROPERTY,
    VectorObject,
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


def file_unchanged(index_name: str, namespace: str, file_name: str, fingerprint: str) -> bool:
    """True if this exact file version is already stored (checked before any parsing or vision call)."""
    index = sanitize_index_name(index_name)
    return (
        db_connection.tenant_exists(index, namespace)
        and db_connection.get_content_hash(index, namespace, file_name) == fingerprint
    )


async def is_file_unchanged(index_name: str, namespace: str, file_name: str, fingerprint: str) -> bool:
    return await asyncio.to_thread(file_unchanged, index_name, namespace, file_name, fingerprint)


def _embed(documents: list[Document], images: list[Optional[bytes]]) -> list[list[float]]:
    """Text chunks through the (cached) text embedder; image chunks from pixels + text
    when the provider is multimodal, otherwise from their text like every other chunk."""
    image_embedder = get_image_embedder()
    image_positions = [i for i, image in enumerate(images) if image is not None] if image_embedder else []
    image_set = set(image_positions)
    text_positions = [i for i in range(len(documents)) if i not in image_set]

    vectors: list[Optional[list[float]]] = [None] * len(documents)
    text_vectors = get_embeddings().embed_documents([documents[i].page_content for i in text_positions])
    for position, vector in zip(text_positions, text_vectors, strict=True):
        vectors[position] = vector
    if image_positions:
        image_vectors = image_embedder.embed_images(
            [(images[i], documents[i].page_content) for i in image_positions]
        )
        for position, vector in zip(image_positions, image_vectors, strict=True):
            vectors[position] = vector
    return vectors


@traceable(
    name="store_multimodal_in_weaviate",
    run_type="tool",
    process_inputs=lambda inputs: {
        "index_name": inputs["index_name"],
        "namespace": inputs["namespace"],
        "chunks": len(inputs["documents"]),
        "images": sum(image is not None for image in inputs["images"]),
    },
)
def store_documents_with_images(
    documents: list[Document],
    images: list[Optional[bytes]],
    index_name: str,
    namespace: str,
    fingerprint: str,
) -> int:
    """Embed and store one file version, then remove the file's older versions.

    ``images[i]`` holds the image bytes for an image chunk, else None. Every
    document needs ``chunk_id`` and ``file_name`` metadata; ``fingerprint`` is
    the version key stored as ``content_hash``.
    New chunks are written first and old versions deleted only afterwards, so a
    failed write never leaves the file missing; a failed write removes its own
    partial chunks. Chunk ids are deterministic, so concurrent uploads of the
    same file overwrite the same objects instead of duplicating them.
    """
    if not documents:
        return 0
    index = db_connection.ensure_collection(index_name)
    file_name = documents[0].metadata[FILE_NAME_PROPERTY]

    vectors = _embed(documents, images)
    objects = [
        VectorObject(
            uuid=doc.metadata["chunk_id"],
            properties={
                config_settings.WEAVIATE_TEXT_KEY: doc.page_content,
                **doc.metadata,
                CONTENT_HASH_PROPERTY: fingerprint,
            },
            vector=vector,
        )
        for doc, vector in zip(documents, vectors, strict=True)
    ]

    try:
        db_connection.insert_objects(index, namespace, objects)
    except VectorStoreError:
        db_connection.delete_file_version(index, namespace, file_name, fingerprint)
        raise

    removed = db_connection.delete_file_versions_except(index, namespace, file_name, keep_hash=fingerprint)
    logger.info(f"Stored {len(objects)} chunks of {file_name} in {index}/{namespace}; removed {removed} old chunks")
    return len(objects)


async def push_documents_with_images(
    documents: list[Document],
    images: list[Optional[bytes]],
    index_name: str,
    namespace: str,
    fingerprint: str,
) -> PushToDatabaseResponseDto:
    """Async wrapper of ``store_documents_with_images``; failures are returned in the response."""
    try:
        stored = await asyncio.to_thread(
            store_documents_with_images, documents, images, index_name, namespace, fingerprint
        )
        return PushToDatabaseResponseDto(status=True, chunks_stored=stored, index=index_name, namespace=namespace)
    except Exception as error:  # noqa: BLE001
        logger.exception(f"Failed to push documents to {index_name}/{namespace}")
        return PushToDatabaseResponseDto(status=False, index=index_name, namespace=namespace, error=str(error))


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
