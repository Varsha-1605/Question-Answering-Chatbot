import asyncio
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from ai_workflow.retrieval.prompts import initialise_title_prompt_template
from ai_workflow.utils import get_rewrite_model

MAX_TITLE_CHARS = 40
MAX_QUESTION_CHARS = 500
TITLE_TIMEOUT_SECONDS = 8


def clean_title(raw: str) -> Optional[str]:
    """Reduce the model output to a short single-line title, or None."""
    lines = raw.strip().splitlines()
    if not lines:
        return None
    title = lines[0].strip().strip("\"'`*#").strip().rstrip(".!?:;,").strip()
    if not title or title.lower() == "none":
        return None
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rstrip() + "…"
    return title


async def generate_chat_title(question: str) -> Optional[str]:
    """Title a chat from its first question; None on any failure."""
    try:
        chain = initialise_title_prompt_template() | get_rewrite_model() | StrOutputParser()
        raw = await asyncio.wait_for(
            chain.ainvoke({"question": question[:MAX_QUESTION_CHARS]}),
            timeout=TITLE_TIMEOUT_SECONDS,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Chat title skipped: {error}")
        return None
    return clean_title(raw)
