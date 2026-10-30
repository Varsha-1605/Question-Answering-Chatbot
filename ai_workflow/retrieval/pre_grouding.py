from typing import Callable, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from ai_workflow.retrieval.prompts import (
    initialise_classify_prompt_template,
    initialise_rewrite_prompt_template,
)
from ai_workflow.utils import get_rewrite_model

Intent = Literal["DOC", "WEB", "NONE"]


async def transform_user_query_for_retrieval(
    question: str,
    chat_history: str = "",
    chat_model_factory: Callable[[], BaseChatModel] = get_rewrite_model,
) -> Optional[str]:
    """Return a search query, or ``None`` for plain conversation (falls back to the raw question)."""
    try:
        chain = initialise_rewrite_prompt_template() | chat_model_factory() | StrOutputParser()
        rewritten = (
            await chain.ainvoke(
                {"question": question,
                 "chat_history": chat_history or "(none)"}
                 )
        ).strip().strip("\"'")
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Query rewrite failed, using the original question: {error}")
        return question

    if not rewritten or rewritten.lower() == "none":
        return None
    return rewritten


async def classify_user_query(
    question: str,
    chat_history: str = "",
    chat_model_factory: Callable[[], BaseChatModel] = get_rewrite_model,
) -> Intent:
    """Classify without rewriting: DOC, WEB or NONE (falls back to DOC)."""
    try:
        chain = initialise_classify_prompt_template() | chat_model_factory() | StrOutputParser()
        label = (
            await chain.ainvoke({"question": question, "chat_history": chat_history or "(none)"})
        ).strip().strip("\"'.").upper()
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Query classification failed, searching the documents: {error}")
        return "DOC"

    if label.startswith("WEB"):
        return "WEB"
    if label.startswith("NONE"):
        return "NONE"
    return "DOC"
