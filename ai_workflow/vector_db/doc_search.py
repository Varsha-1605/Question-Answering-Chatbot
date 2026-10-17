import asyncio
from typing import Optional

from langchain_core.documents import Document
from langsmith import traceable
from loguru import logger

from ai_workflow.exception import DocumentRetrievalError
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.sanitization import sanitize_namespace
from ai_workflow.vector_db.utils import get_vector_store
from ai_workflow.vector_db.weaviate_utils.weaviate_client import db_connection


@traceable(
    name="hybrid_search",
    run_type="retriever",
    process_inputs=lambda inputs: {
        "query": inputs["question"],
        "index_name": inputs["index_name"],
        "namespace": inputs["namespace"],
        "top_k": inputs["top_k"],
        "alpha": config_settings.WEAVIATE_HYBRID_ALPHA,
    },
    process_outputs=lambda results: {
        "documents": [
            {
                "page_content": document.page_content,
                "type": "Document",
                "metadata": {**document.metadata, "score": score},
            }
            for document, score in results
        ]
    },
)
def _hybrid_search(
    question: str, index_name: str, namespace: str, top_k: int
) -> list[tuple[Document, float]]:
    """Hybrid search (blocking); a missing namespace returns no results."""
    if not db_connection.tenant_exists(index_name, namespace):
        logger.info(f"No namespace {namespace!r} in index {index_name!r}; nothing to search")
        return []

    try:
        results = get_vector_store(index_name).similarity_search_with_score(
            question,
            k=top_k,
            tenant=sanitize_namespace(namespace),
            alpha=config_settings.WEAVIATE_HYBRID_ALPHA,
        )
    except Exception:
        # The namespace may have been deleted elsewhere; recheck next time.
        db_connection.forget_tenant(index_name, namespace)
        raise

    # Drop results without a score.
    scored = [(doc, score) for doc, score in results if score is not None]
    if len(scored) != len(results):
        logger.warning(f"Dropped {len(results) - len(scored)} results that had no score")
    return scored


async def get_related_docs_with_score(
    question: str,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    total_docs_to_retrieve: Optional[int] = None,
) -> list[tuple[Document, float]]:
    """Return ``(document, score)`` pairs, best first; raises ``DocumentRetrievalError`` on failure."""
    index_name = index_name or config_settings.INDEX_NAME
    namespace = namespace or config_settings.DEFAULT_NAMESPACE
    top_k = total_docs_to_retrieve or config_settings.RETRIEVAL_TOP_K

    try:
        results = await asyncio.to_thread(
            _hybrid_search, question, index_name, namespace, top_k
        )
    except Exception as error:  # noqa: BLE001
        logger.exception(f"Vector search failed for {index_name}/{namespace}")
        raise DocumentRetrievalError(f"Vector search failed: {error}") from error

    logger.info(f"Retrieved {len(results)} chunks from {index_name}/{namespace}")
    return results

