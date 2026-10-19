import asyncio
import re
import time
import uuid
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket
from langchain_core.messages import AIMessageChunk
from loguru import logger
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from app.exception import DocumentRetrievalError, RAGError
from app.retrieval.answer_cache import get_answer_cache
from app.retrieval.chat_memory import chat_memory_enabled, get_graph, load_history, record_turn
from app.retrieval.initialize_memory import format_chat_history
from app.retrieval.models import ChatRequest, Message, TitleRequest
from app.retrieval.rag_util import ConnectionManager, send_message_over_websocket
from app.retrieval.title import generate_chat_title
from app.settings import config_settings
from app.vector_db.utils import has_documents

router = APIRouter(tags=["chat"])
manager = ConnectionManager()

WS_NORMAL_CLOSE = 1000
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011

# Graph nodes whose output is streamed to the user.
ANSWER_NODES = ("converse", "answer")

TRACE_RUN_NAME = "Conversation"

# Cached answers are streamed word by word, like generated ones.
_STREAM_PIECE = re.compile(r"\S+\s*")


async def _emit(
        token: str, websocket: Optional[WebSocket], on_token: Optional[Callable[[str], None]]
) -> None:
    if on_token is not None:
        on_token(token)
    if websocket is not None:
        await send_message_over_websocket(websocket, token, "stream", "answer")


async def run_rag(
        question: str,
        language: str,
        chat_context: list[Message],
        websocket: Optional[WebSocket],
        index_name: str,
        namespace: str,
        on_token: Optional[Callable[[str], None]] = None,
        thread_id: Optional[str] = None,
) -> str:
    """Answer a question from the documents; tokens go to ``websocket`` and ``on_token``."""
    thread_id = thread_id or str(uuid.uuid4())
    graph_config = {"run_name": TRACE_RUN_NAME, "configurable": {"thread_id": thread_id}}
    turn_started = time.perf_counter()
    logger.info(f"User query: {question}")

    # With Postgres memory, the stored history replaces the client's.
    if chat_memory_enabled():
        chat_context = await load_history(thread_id)

    graph_input = {
        "question": question,
        "chat_history": format_chat_history(chat_context),
        "language": language,
        "index_name": index_name,
        "namespace": namespace,
    }

    answer_cache = get_answer_cache()
    cache_probe = None
    if answer_cache is not None:
        cache_probe = await answer_cache.probe(
            question=question,
            has_history=bool(graph_input["chat_history"]),
            language=language,
            index_name=index_name,
            namespace=namespace,
        )

    if cache_probe is not None and cache_probe.hit is not None:
        hit = cache_probe.hit
        logger.info(f"Semantic cache hit (similarity {hit.similarity:.3f}, cached question {hit.question!r})")
        for piece in _STREAM_PIECE.findall(hit.answer):
            await _emit(piece, websocket, on_token)
        answer = hit.answer
        await record_turn(thread_id, question, answer)
        logger.info(f"Total response time: {time.perf_counter() - turn_started:.2f}s (cached)")
    else:
        answer, passages_used = await _generate_answer(
            graph_input, graph_config, websocket, on_token, turn_started
        )
        logger.info(f"Total response time: {time.perf_counter() - turn_started:.2f}s")
        # Only answers based on retrieved passages are cached.
        if cache_probe is not None and passages_used:
            await answer_cache.save(cache_probe, answer)

    if websocket is not None:
        await send_message_over_websocket(websocket, "", "end")
    return answer


async def _generate_answer(
        graph_input: dict,
        graph_config: dict,
        websocket: Optional[WebSocket],
        on_token: Optional[Callable[[str], None]],
        turn_started: float,
) -> tuple[str, int]:
    """Run the graph, stream the answer; returns (answer, passage count)."""
    answer = ""
    passages_used = 0
    try:
        async for mode, payload in get_graph().astream(
                graph_input, config=graph_config, stream_mode=["updates", "messages"]
        ):
            if mode == "updates":
                if "transform_query" in payload:
                    retrieval_query = payload["transform_query"]["retrieval_query"]
                    logger.info(
                        f"Transformed query: {retrieval_query or 'None (conversation, no search)'}"
                    )
                for node in ("decide", "search"):
                    if "passages" in (payload.get(node) or {}):
                        passages_used = len(payload[node]["passages"])
                continue

            chunk, metadata = payload
            # Keep only streamed tokens, not whole checkpointed messages.
            if (
                not isinstance(chunk, AIMessageChunk)
                or metadata.get("langgraph_node") not in ANSWER_NODES
                or not chunk.content
            ):
                continue

            if not answer:
                logger.info(f"TTFT: {time.perf_counter() - turn_started:.2f}s")
            answer += chunk.content
            await _emit(chunk.content, websocket, on_token)
    except (WebSocketDisconnect, DocumentRetrievalError):
        raise
    except Exception as e:
        logger.exception("Answer generation failed")
        raise RAGError(f"Answer generation failed: {e}") from e
    return answer, passages_used


async def refuse(websocket: WebSocket, message: str, code: str, close_code: int) -> None:
    await send_message_over_websocket(websocket, message, "error", code=code)
    await websocket.close(code=close_code)


@router.get("/chat/threads/{thread_id}/messages")
async def thread_messages(thread_id: uuid.UUID) -> dict:
    """Stored messages of a thread."""
    return {"messages": [m.model_dump() for m in await load_history(str(thread_id))]}


@router.post("/chat/title")
async def chat_title(request: TitleRequest) -> dict:
    """Short title for a chat's first question (null if it could not be made)."""
    return {"title": await generate_chat_title(request.question)}


@router.websocket("/ws/chat")
@router.websocket("/ws/chat/{index_name}/{namespace}")
async def websocket_run_rag(
        websocket: WebSocket,
        index_name: str = config_settings.INDEX_NAME,
        namespace: str = config_settings.DEFAULT_NAMESPACE,
        thread_id: Optional[uuid.UUID] = None,
) -> None:
    await websocket.accept()
    # One conversation per connection; ?thread_id= resumes an old one.
    thread_id = str(thread_id or uuid.uuid4())

    if not manager.try_connect(websocket):
        await refuse(
            websocket,
            f"Server at maximum capacity ({manager.max_connections} connections). Try again later.",
            "MAX_CONNECTIONS_REACHED",
            WS_POLICY_VIOLATION,
        )
        return

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=config_settings.WEBSOCKET_IDLE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                await send_message_over_websocket(
                    websocket, "Connection closed due to inactivity.", "error", code="IDLE_TIMEOUT"
                )
                await websocket.close(code=WS_NORMAL_CLOSE)
                return

            if message.strip().lower() == "ping":
                await websocket.send_text("pong")
                continue

            try:
                request = ChatRequest.model_validate_json(message)
            except (ValidationError, ValueError) as e:
                await send_message_over_websocket(
                    websocket, f"Invalid request: {e}", "error", code="INVALID_REQUEST"
                )
                continue

            try:
                await run_rag(
                    question=request.question,
                    language=request.language,
                    chat_context=request.chat_context,
                    websocket=websocket,
                    index_name=index_name,
                    namespace=namespace,
                    thread_id=thread_id,
                )
            except DocumentRetrievalError:
                await send_message_over_websocket(
                    websocket, "Document search is temporarily unavailable.", "error", code="SEARCH_FAILED"
                )
            except RAGError:
                await send_message_over_websocket(
                    websocket, "The answer could not be generated. Please retry.", "error", code="ANSWER_FAILED"
                )
    except WebSocketDisconnect:
        logger.info("Chat client disconnected")
    finally:
        manager.disconnect(websocket)


if __name__ == "__main__":
    # Local chat: python -m app.retrieval.routes_with_graph
    EXIT_WORDS = {"bye", "exit", "quit"}

    async def main():
        from app.vector_db.utils import close_vector_db
        from app.warm_up import warm_up

        index_name = config_settings.INDEX_NAME
        namespace = config_settings.DEFAULT_NAMESPACE
        history: list[Message] = []
        thread_id = str(uuid.uuid4())
        try:
            if not await has_documents(index_name, namespace):
                raise SystemExit(
                    f"Nothing ingested in {index_name}/{namespace}. "
                    "Run: python -m app.injestion.utils"
                )

            print("Warming up (network, Weaviate, chat model, embeddings)...")
            started = time.perf_counter()
            await warm_up()
            print(f"Warm-up took {time.perf_counter() - started:.2f}s\n")
            print(f"Ask about the documents in {index_name}/{namespace}. Type bye, exit or quit to stop.")

            while True:
                try:
                    question = (await asyncio.to_thread(input, "\nYou: ")).strip()
                except EOFError:
                    break
                if not question:
                    continue
                if question.lower() in EXIT_WORDS:
                    break

                started = time.perf_counter()
                first_token_at = None

                def show(token: str) -> None:
                    nonlocal first_token_at
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - started
                        print("\nBot: ", end="", flush=True)
                    print(token, end="", flush=True)

                answer = await run_rag(
                    question=question, language="English", chat_context=history,
                    websocket=None, index_name=index_name, namespace=namespace,
                    on_token=show, thread_id=thread_id,
                )
                total = time.perf_counter() - started
                print(f"\n[first word after {first_token_at or total:.2f}s, finished in {total:.2f}s]")

                history += [Message(type="human", content=question), Message(type="ai", content=answer)]
            print("Bye.")
        finally:
            close_vector_db()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")