import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from app.exception import DocumentRetrievalError, RAGError
from app.retrieval import routes_with_graph as chat
from app.retrieval.answer_cache import CacheProbe
from app.cache.semantic import CachedAnswer


def token(text, node="answer"):
    return "messages", (AIMessageChunk(content=text), {"langgraph_node": node})


def passages(count):
    return "updates", {"decide": {"passages": [object()] * count}}


class FakeGraph:
    def __init__(self, events, error=None):
        self.events, self.error, self.runs = events, error, 0

    async def astream(self, graph_input, config, stream_mode):
        self.runs += 1
        for event in self.events:
            yield event
        if self.error:
            raise self.error


class FakeCache:
    def __init__(self, hit=None):
        self.hit, self.saved = hit, []

    async def probe(self, **kwargs):
        return CacheProbe(scope="s", question="q", vector=[0.1], hit=self.hit)

    async def save(self, probe, answer):
        self.saved.append(answer)


@pytest.fixture
def wire(monkeypatch):
    class Wire:
        recorded = []
        cache = None

    async def record_turn(thread_id, question, answer):
        Wire.recorded.append((question, answer))

    Wire.recorded, Wire.cache = [], None
    monkeypatch.setattr(chat, "chat_memory_enabled", lambda: False)
    monkeypatch.setattr(chat, "record_turn", record_turn)
    monkeypatch.setattr(chat, "get_answer_cache", lambda: Wire.cache)
    monkeypatch.setattr(chat, "get_graph", lambda: Wire.graph)
    return Wire


def run(question="What is the yield?", **kwargs):
    tokens = []
    answer = asyncio.run(
        chat.run_rag(
            question=question,
            language="English",
            chat_context=[],
            websocket=None,
            index_name="Idx",
            namespace="ns",
            on_token=tokens.append,
            **kwargs,
        )
    )
    return answer, tokens


def test_only_tokens_from_the_answer_nodes_are_streamed(wire):
    wire.graph = FakeGraph([token("rewritten", node="transform_query"), token("Hel"), token("lo", node="converse")])
    answer, tokens = run()
    assert (answer, tokens) == ("Hello", ["Hel", "lo"])


def test_empty_and_non_chunk_messages_are_not_tokens(wire):
    wire.graph = FakeGraph([token(""), ("messages", ("a whole message", {"langgraph_node": "answer"})), token("ok")])
    assert run()[0] == "ok"


def test_answer_built_from_passages_is_saved_to_the_cache(wire):
    wire.cache = FakeCache()
    wire.graph = FakeGraph([passages(2), token("5 t/ha")])
    run()
    assert wire.cache.saved == ["5 t/ha"]


def test_answer_without_passages_is_not_cached(wire):
    wire.cache = FakeCache()
    wire.graph = FakeGraph([token("Hi!", node="converse")])
    run("hello")
    assert wire.cache.saved == []


def test_cache_hit_streams_the_stored_answer_without_running_the_graph(wire):
    wire.cache = FakeCache(hit=CachedAnswer(question="q", answer="cached answer here", similarity=0.99))
    wire.graph = FakeGraph([token("never")])
    answer, tokens = run()
    assert answer == "cached answer here"
    assert "".join(tokens) == "cached answer here"
    assert wire.graph.runs == 0
    assert wire.recorded == [("What is the yield?", "cached answer here")]
    assert wire.cache.saved == []


def test_generation_failure_becomes_rag_error(wire):
    wire.graph = FakeGraph([], error=RuntimeError("llm down"))
    with pytest.raises(RAGError):
        run()


def test_search_failure_keeps_its_own_error_type(wire):
    wire.graph = FakeGraph([], error=DocumentRetrievalError("weaviate down"))
    with pytest.raises(DocumentRetrievalError):
        run()
