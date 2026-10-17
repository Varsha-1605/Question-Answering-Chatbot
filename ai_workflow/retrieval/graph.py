from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from ai_workflow.retrieval.pre_grouding import transform_user_query_for_retrieval
from ai_workflow.retrieval.prompts import (
    initialise_conversation_prompt_template,
    initialise_doc_search_prompt_template,
)
from ai_workflow.retrieval.question_rules import is_standalone
from ai_workflow.retrieval.reranker import RerankingRetriever
from ai_workflow.utils import get_chat_model

Passage = tuple[Document, float]
NextStep = Literal["converse", "answer", "search"]


class RagState(TypedDict, total=False):
    question: str
    chat_history: str
    language: str
    index_name: str
    namespace: str
    messages: Annotated[list[AnyMessage], add_messages]
    retrieval_query: Optional[str]
    raw_passages: Optional[list[Passage]]
    next_step: NextStep
    passages: list[Passage]
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


async def transform_query(state: RagState) -> dict:
    query = await transform_user_query_for_retrieval(
        question=state["question"], chat_history=state["chat_history"]
    )
    return {"retrieval_query": query}


async def raw_search(state: RagState) -> dict:
    """Search with the raw question, in parallel with the rewrite."""
    try:
        return {"raw_passages": await _retrieve(state, search_query=state["question"])}
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Raw-question search failed, will search after the rewrite: {error}")
        return {"raw_passages": None}


def decide(state: RagState) -> dict:
    """Choose the next step after both parallel nodes finish."""
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

    # Both start together; decide waits for both.
    graph.add_edge(START, "transform_query")
    graph.add_edge(START, "raw_search")
    graph.add_edge(["transform_query", "raw_search"], "decide")

    graph.add_conditional_edges(
        "decide", route, {"converse": "converse", "answer": "answer", "search": "search"}
    )
    graph.add_edge("search", "answer")
    graph.add_edge("converse", END)
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)


rag_graph = build_rag_graph()