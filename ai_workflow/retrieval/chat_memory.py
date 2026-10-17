import asyncio
import sys
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ai_workflow.retrieval.graph import build_rag_graph, rag_graph
from ai_workflow.retrieval.models import Message
from ai_workflow.settings import config_settings

_pool: Optional[AsyncConnectionPool] = None
_graph = None


def chat_memory_enabled() -> bool:
    return _graph is not None


def get_graph():
    """Postgres-backed graph, or the stateless one when memory is off."""
    return _graph or rag_graph


async def open_chat_memory() -> None:
    """Connect to Postgres and set up checkpoint tables (needs POSTGRES_URL)."""
    global _pool, _graph
    if not config_settings.POSTGRES_URL or _graph is not None:
        return

    # psycopg async does not work on the Windows Proactor loop.
    if sys.platform == "win32" and isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
        raise RuntimeError(
            "Postgres chat memory needs a selector event loop on Windows: "
            "start with `uvicorn main:app --loop asyncio:SelectorEventLoop`"
        )

    pool = AsyncConnectionPool(
        conninfo=config_settings.POSTGRES_URL.get_secret_value(),
        max_size=config_settings.POSTGRES_POOL_MAX_SIZE,
        # Settings AsyncPostgresSaver requires.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open(wait=True)
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
    except Exception:
        await pool.close()
        raise

    _pool, _graph = pool, build_rag_graph(checkpointer=saver)
    logger.info("Chat memory on: conversations are checkpointed in Postgres")


async def close_chat_memory() -> None:
    global _pool, _graph
    if _pool is not None:
        await _pool.close()
    _pool = _graph = None


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def load_history(thread_id: str) -> list[Message]:
    """Stored messages of a thread, oldest first."""
    if _graph is None:
        return []
    snapshot = await _graph.aget_state(thread_config(thread_id))
    return [
        Message(type="human" if isinstance(m, HumanMessage) else "ai", content=str(m.content))
        for m in snapshot.values.get("messages", [])
        if isinstance(m, (HumanMessage, AIMessage))
    ]


async def record_turn(thread_id: str, question: str, answer: str) -> None:
    """Save a turn that skipped the graph (cache hit)."""
    if _graph is None:
        return
    await _graph.aupdate_state(
        thread_config(thread_id),
        {"messages": [HumanMessage(question), AIMessage(answer)]},
        as_node="answer",
    )
