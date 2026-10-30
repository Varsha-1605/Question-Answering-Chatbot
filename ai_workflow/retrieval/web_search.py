from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from langsmith import traceable
from loguru import logger
from tavily import AsyncTavilyClient

from ai_workflow.settings import config_settings

Passage = tuple[Document, float]


def web_search_enabled() -> bool:
    return config_settings.TAVILY_API_KEY is not None


class WebSearch:
    """Tavily web search; results come back as passages like the document retriever's."""

    def __init__(self, api_key: str, max_results: int, timeout: float) -> None:
        self.client = AsyncTavilyClient(api_key=api_key)
        self.max_results = max_results
        self.timeout = timeout

    @classmethod
    @lru_cache(maxsize=1)
    def from_settings(cls) -> Optional["WebSearch"]:
        """The search tool (built once: creating the client takes ~0.3s), or None without a TAVILY_API_KEY."""
        if not web_search_enabled():
            return None
        return cls(
            api_key=config_settings.TAVILY_API_KEY.get_secret_value(),
            max_results=config_settings.WEB_SEARCH_MAX_RESULTS,
            timeout=config_settings.WEB_SEARCH_TIMEOUT_SECONDS,
        )

    @traceable(
        name="web_search",
        run_type="tool",
        process_inputs=lambda inputs: {"query": inputs["query"]},
        process_outputs=lambda passages: {
            "results": [document.metadata.get("url") for document, _ in passages]
        },
    )
    async def search(self, query: str) -> list[Passage]:
        """Return web results as passages; an empty list if the search fails."""
        logger.info(f"Web search tool called: {query!r}")
        try:
            response = await self.client.search(
                query=query, max_results=self.max_results, timeout=self.timeout
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Web search failed: {error}")
            return []

        passages = [
            (
                Document(
                    page_content=result.get("content", ""),
                    metadata={"file_name": result.get("title") or "web", "url": result.get("url")},
                ),
                float(result.get("score") or 0.0),
            )
            for result in response.get("results", [])
            if result.get("content")
        ]
        logger.info(f"Web search returned {len(passages)} results")
        return passages
