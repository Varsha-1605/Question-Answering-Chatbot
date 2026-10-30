from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from ai_workflow.retrieval.pre_grouding import (
    classify_user_query,
    transform_user_query_for_retrieval,
)
from ai_workflow.retrieval.prompts import (
    initialise_conversation_prompt_template,
    initialise_doc_search_prompt_template,
    initialise_web_search_prompt_template,
    not_found_reply,
)
from ai_workflow.retrieval.question_rules import is_small_talk, is_standalone
from ai_workflow.retrieval.reranker import RerankingRetriever
from ai_workflow.retrieval.web_search import WebSearch
from ai_workflow.settings import config_settings
from ai_workflow.utils import get_chat_model

Passage = tuple[Document, float]
NextStep = Literal["converse", "answer", "search", "web_search"]

# The rewrite step marks current-events questions with this prefix.
WEB_PREFIX = "WEB:"


class RagState(TypedDict, total=False):
    question: str
    chat_history: str
    language: str
    index_name: str
    namespace: str
    messages: Annotated[list[AnyMessage], add_messages]
    retrieval_query: Optional[str]
    web_query: Optional[str]
    raw_passages: Optional[list[Passage]]
    next_step: NextStep
    passages: list[Passage]
    web_passages: list[Passage]
    answer: str


def format_context(passages: list[Passage]) -> str:
    """Format passages as numbered, labelled blocks."""
    blocks = []
    for number, (doc, _score) in enumerate(passages, start=1):
        label = doc.metadata.get("file_name", "unknown")
        if doc.metadata.get("page") is not None:
            label += f", page {doc.metadata['page']}"
        blocks.append(f"[{number}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_web_context(passages: list[Passage]) -> str:
    """Format web results as numbered blocks labelled with title and URL."""
    return "\n\n".join(
        f"[{number}] ({doc.metadata.get('file_name', 'web')}, {doc.metadata.get('url', 'no url')})\n"
        f"{doc.page_content}"
        for number, (doc, _score) in enumerate(passages, start=1)
    )


def _turn(state: RagState, answer: str) -> list[AnyMessage]:
    """The question and answer messages."""
    return [HumanMessage(state["question"]), AIMessage(answer)]


async def _retrieve(state: RagState, search_query: str) -> list[Passage]:
    return await RerankingRetriever.from_settings().retrieve(
        question=state["question"],
        search_query=search_query,
        index_name=state["index_name"],
        namespace=state["namespace"],
    )


async def _classify_or_rewrite(state: RagState) -> Optional[str]:
    """Search query, "WEB: <query>" or None. Standalone questions are only classified."""
    question, history = state["question"], state["chat_history"]
    if is_small_talk(question):
        logger.info("Small talk: classifier skipped")
        return None
    if not config_settings.REWRITE_STANDALONE_QUESTIONS and is_standalone(question, bool(history)):
        intent = await classify_user_query(question=question, chat_history=history)
        return {"DOC": question, "WEB": f"{WEB_PREFIX} {question}", "NONE": None}[intent]
    return await transform_user_query_for_retrieval(question=question, chat_history=history)


async def transform_query(state: RagState) -> dict:
    query = await _classify_or_rewrite(state)
    if query and query.upper().startswith(WEB_PREFIX):
        web_query = query[len(WEB_PREFIX):].strip()
        # Without a Tavily key (or a query) this is plain conversation, as before.
        if web_query and WebSearch.from_settings() is not None:
            return {"retrieval_query": None, "web_query": web_query}
        return {"retrieval_query": None, "web_query": None}
    return {"retrieval_query": query, "web_query": None}


async def raw_search(state: RagState) -> dict:
    """Search with the raw question, in parallel with the rewrite."""
    if is_small_talk(state["question"]):
        return {"raw_passages": None}
    try:
        return {"raw_passages": await _retrieve(state, search_query=state["question"])}
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Raw-question search failed, will search after the rewrite: {error}")
        return {"raw_passages": None}


def decide(state: RagState) -> dict:
    """Choose the next step after both parallel nodes finish."""
    if state.get("web_query"):
        return {"next_step": "web_search"}
    if state["retrieval_query"] is None:
        return {"next_step": "converse"}

    standalone = is_standalone(state["question"], has_history=bool(state["chat_history"]))
    if standalone and state["raw_passages"] is not None:
        return {"next_step": "answer", "passages": state["raw_passages"]}
    return {"next_step": "search"}


def route(state: RagState) -> NextStep:
    return state["next_step"]


async def converse(state: RagState) -> dict:
    """Reply directly without searching."""
    chain = initialise_conversation_prompt_template() | get_chat_model() | StrOutputParser()
    answer = await chain.ainvoke(
        {
            "question": state["question"],
            "chat_history": state["chat_history"] or "(none)",
            "language": state["language"],
        }
    )
    return {"answer": answer, "messages": _turn(state, answer)}


async def search(state: RagState) -> dict:
    """Search with the rewritten query."""
    return {"passages": await _retrieve(state, search_query=state["retrieval_query"])}


async def answer(state: RagState) -> dict:
    """Answer from the retrieved passages."""
    passages = state["passages"]
    chain = initialise_doc_search_prompt_template() | get_chat_model() | StrOutputParser()
    reply = await chain.ainvoke(
        {
            "question": state["question"],
            "chat_history": state["chat_history"] or "(none)",
            "language": state["language"],
            "doc_count": len(passages),
            "context": format_context(passages) or "(no passages found)",
            "not_found_reply": not_found_reply(len(passages)),
        }
    )
    return {"answer": reply, "messages": _turn(state, reply)}


async def web_search(state: RagState) -> dict:
    """Search the web for current-events questions."""
    tool = WebSearch.from_settings()
    results = await tool.search(state["web_query"]) if tool is not None else []
    return {"web_passages": results}


def route_web(state: RagState) -> Literal["web_answer", "converse"]:
    """Answer from web results, or just talk if the search found nothing."""
    return "web_answer" if state["web_passages"] else "converse"


async def web_answer(state: RagState) -> dict:
    """Answer from the web results."""
    passages = state["web_passages"]
    chain = initialise_web_search_prompt_template() | get_chat_model() | StrOutputParser()
    reply = await chain.ainvoke(
        {
            "question": state["question"],
            "chat_history": state["chat_history"] or "(none)",
            "language": state["language"],
            "doc_count": len(passages),
            "context": format_web_context(passages),
        }
    )
    return {"answer": reply, "messages": _turn(state, reply)}


def build_rag_graph(checkpointer=None):
    graph = StateGraph(RagState)
    graph.add_node("transform_query", transform_query)
    graph.add_node("raw_search", raw_search)
    graph.add_node("decide", decide)
    graph.add_node("converse", converse)
    graph.add_node("search", search)
    graph.add_node("answer", answer)
    graph.add_node("web_search", web_search)
    graph.add_node("web_answer", web_answer)

    # Both start together; decide waits for both.
    graph.add_edge(START, "transform_query")
    graph.add_edge(START, "raw_search")
    graph.add_edge(["transform_query", "raw_search"], "decide")

    graph.add_conditional_edges(
        "decide",
        route,
        {"converse": "converse", "answer": "answer", "search": "search", "web_search": "web_search"},
    )
    graph.add_conditional_edges(
        "web_search", route_web, {"web_answer": "web_answer", "converse": "converse"}
    )
    graph.add_edge("search", "answer")
    graph.add_edge("web_answer", END)
    graph.add_edge("converse", END)
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)


rag_graph = build_rag_graph()