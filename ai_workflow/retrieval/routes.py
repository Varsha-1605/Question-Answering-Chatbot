import asyncio
import time
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from ai_workflow.exception import DocumentRetrievalError, RAGError
from ai_workflow.retrieval.graph import format_context
from ai_workflow.retrieval.initialize_memory import format_chat_history
from ai_workflow.retrieval.models import ChatRequest, Message
from ai_workflow.retrieval.pre_grouding import transform_user_query_for_retrieval
from ai_workflow.retrieval.prompts import (
    initialise_conversation_prompt_template,
    initialise_doc_search_prompt_template,
)
from ai_workflow.retrieval.rag_util import ConnectionManager, send_message_over_websocket
from ai_workflow.retrieval.reranker import RerankingRetriever
from ai_workflow.settings import config_settings
from ai_workflow.utils import get_chat_model
from ai_workflow.vector_db.utils import has_documents

router = APIRouter(tags=["chat"])
manager = ConnectionManager()

WS_NORMAL_CLOSE = 1000
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011


async def run_rag(
        question: str,
        language: str,
        chat_context: list[Message],
        websocket: Optional[WebSocket],
        index_name: str,
        namespace: str,
        on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Answer a question from the documents; tokens go to ``websocket`` and ``on_token``."""
    turn_started = time.perf_counter()
    logger.info(f"User query: {question}")
    chat_history = format_chat_history(chat_context)

    # None means small talk.
    retrieval_query = await transform_user_query_for_retrieval(
        question=question, chat_history=chat_history
    )
    logger.info(f"Transformed query: {retrieval_query or 'None (conversation, no search)'}")

    prompt_inputs = {
        "question": question,
        "chat_history": chat_history or "(none)",
        "language": language,
    }

    if retrieval_query is None:
        prompt = initialise_conversation_prompt_template()
    else:
        passages = await RerankingRetriever.from_settings().retrieve(
            question=question,
            search_query=retrieval_query,
            index_name=index_name,
            namespace=namespace,
        )

        prompt = initialise_doc_search_prompt_template()
        prompt_inputs["doc_count"] = len(passages)
        prompt_inputs["context"] = format_context(passages) or "(no passages found)"

    llm_chain = prompt | get_chat_model() | StrOutputParser()

    answer = ""
    try:
        async for token in llm_chain.astream(prompt_inputs):
            if token:
                if not answer:
                    logger.info(f"TTFT: {time.perf_counter() - turn_started:.2f}s")
                answer += token
                if on_token is not None:
                    on_token(token)
                if websocket is not None:
                    await send_message_over_websocket(websocket, token, "stream", "answer")
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.exception("Answer generation failed")
        raise RAGError(f"Answer generation failed: {e}") from e

    logger.info(f"Total response time: {time.perf_counter() - turn_started:.2f}s")

    if websocket is not None:
        await send_message_over_websocket(websocket, "", "end")
    return answer


async def refuse(websocket: WebSocket, message: str, code: str, close_code: int) -> None:
    await send_message_over_websocket(websocket, message, "error", code=code)
    await websocket.close(code=close_code)


@router.websocket("/ws/chat")
@router.websocket("/ws/chat/{index_name}/{namespace}")
async def websocket_run_rag(
        websocket: WebSocket,
        index_name: str = config_settings.INDEX_NAME,
        namespace: str = config_settings.DEFAULT_NAMESPACE,
) -> None:
    await websocket.accept()

    if not manager.try_connect(websocket):
        await refuse(
            websocket,
            f"Server at maximum capacity ({manager.max_connections} connections). Try again later.",
            "MAX_CONNECTIONS_REACHED",
            WS_POLICY_VIOLATION,
        )
        return

    try:
        # Chat only if the namespace has documents.
        try:
            if not await has_documents(index_name, namespace):
                await refuse(
                    websocket,
                    "No ingested documents found. Ingest files first.",
                    "NO_DOCUMENTS",
                    WS_POLICY_VIOLATION,
                )
                return
        except ValueError:
            await refuse(websocket, "Invalid index or namespace name.", "INVALID_TARGET", WS_POLICY_VIOLATION)
            return
        except Exception:
            logger.exception("Could not check documents before chat")
            await refuse(websocket, "Document store is temporarily unavailable.", "UNAVAILABLE", WS_INTERNAL_ERROR)
            return

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
    # Local chat: python -m ai_workflow.retrieval.routes
    import time

    EXIT_WORDS = {"bye", "exit", "quit"}

    async def main():
        from ai_workflow.vector_db.utils import close_vector_db
        from ai_workflow.warm_up import warm_up

        index_name = config_settings.INDEX_NAME
        namespace = config_settings.DEFAULT_NAMESPACE
        history: list[Message] = []
        try:
            if not await has_documents(index_name, namespace):
                raise SystemExit(
                    f"Nothing ingested in {index_name}/{namespace}. "
                    "Run: python -m ai_workflow.injestion.utils"
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
                    on_token=show,
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
