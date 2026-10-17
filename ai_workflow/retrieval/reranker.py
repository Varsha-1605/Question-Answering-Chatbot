import asyncio
from functools import lru_cache

from langchain_core.documents import Document
from langsmith import traceable
from loguru import logger

from ai_workflow.exception import DocumentRetrievalError
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.doc_search import get_related_docs_with_score

Passage = tuple[Document, float]


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once (imported lazily to avoid loading torch)."""
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading reranker {config_settings.RERANKER_MODEL}")
    return CrossEncoder(config_settings.RERANKER_MODEL)


class RerankingRetriever:
    def __init__(
        self,
        fetch_k: int,
        top_k: int,
        min_score: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.min_score = min_score
        self.enabled = enabled

    @classmethod
    def from_settings(cls) -> "RerankingRetriever":
        return cls(
            fetch_k=max(config_settings.RERANK_CANDIDATES, config_settings.RETRIEVAL_TOP_K),
            top_k=config_settings.RETRIEVAL_TOP_K,
            min_score=config_settings.MIN_RELEVANCE_SCORE,
            enabled=config_settings.RERANKER_ENABLED,
        )

    async def retrieve(
        self, question: str, search_query: str, index_name: str, namespace: str
    ) -> list[Passage]:
        """Return the best ``top_k`` passages (search uses ``search_query``, reranking uses ``question``)."""
        candidates = await get_related_docs_with_score(
            question=search_query,
            index_name=index_name,
            namespace=namespace,
            total_docs_to_retrieve=self.fetch_k if self.enabled else self.top_k,
        )

        # Threshold on the hybrid score, not the reranker's raw logits.
        kept = [(doc, score) for doc, score in candidates if score >= self.min_score]

        passages = await self.rerank(question, kept) if self.enabled else kept

        logger.info(
            f"Using {len(passages)} of {len(candidates)} retrieved passages"
            f"{' (reranked)' if self.enabled else ''}"
        )
        return passages

    @traceable(
        name="rerank",
        run_type="tool",
        process_inputs=lambda inputs: {
            "question": inputs["question"],
            "candidates": len(inputs["passages"]),
            "model": config_settings.RERANKER_MODEL,
        },
        process_outputs=lambda passages: {
            "passages": [
                {
                    "file_name": document.metadata.get("file_name"),
                    "page": document.metadata.get("page"),
                    "score": score,
                }
                for document, score in passages
            ]
        },
    )
    async def rerank(self, question: str, passages: list[Passage]) -> list[Passage]:
        """Return the ``top_k`` passages ranked by cross-encoder score."""
        if not passages:
            return []

        try:
            return await asyncio.to_thread(self._score, question, passages)
        except Exception as error:  # noqa: BLE001
            logger.exception("Reranking failed")
            raise DocumentRetrievalError(f"Reranking failed: {error}") from error

    def _score(self, question: str, passages: list[Passage]) -> list[Passage]:
        """Score and sort passages (blocking)."""
        scores = get_reranker().predict([(question, doc.page_content) for doc, _ in passages])
        ranked = sorted(
            ((doc, float(score)) for (doc, _), score in zip(passages, scores)),
            key=lambda passage: passage[1],
            reverse=True,
        )
        return ranked[: self.top_k]
