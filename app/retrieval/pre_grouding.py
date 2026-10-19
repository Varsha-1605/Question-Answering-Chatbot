from typing import Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.retrieval.prompts import initialise_rewrite_prompt_template
from app.utils import get_rewrite_model


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
