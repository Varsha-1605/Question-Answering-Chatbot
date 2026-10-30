import asyncio
import time

import pytest
from langchain_core.messages import AIMessageChunk

from ai_workflow.exception import DocumentRetrievalError, RAGError
from ai_workflow.retrieval import routes_with_graph as chat
from ai_workflow.retrieval.answer_cache import CacheProbe
from ai_workflow.cache.semantic import CachedAnswer


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
    def __init__(self, hit=None, delay=0.0):
        self.hit, self.saved, self.delay, self.probes = hit, [], delay, 0

    async def probe(self, **kwargs):
        self.probes += 1
        await asyncio.sleep(self.delay)
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


HIT = CachedAnswer(question="q", answer="cached answer here", similarity=0.99)


def test_cache_hit_streams_the_stored_answer_and_none_of_the_graph(wire):
    wire.cache = FakeCache(hit=HIT, delay=0.05)  # the lookup finishes after the graph's first token
    wire.graph = FakeGraph([token("never"), token(" shown")])
    answer, tokens = run()
    assert answer == "cached answer here"
    assert "".join(tokens) == "cached answer here"
    assert wire.recorded == [("What is the yield?", "cached answer here")]
    assert wire.cache.saved == []


def test_cache_hit_shows_no_tool_call(wire):
    wire.cache = FakeCache(hit=HIT)
    wire.graph = FakeGraph(web_turn([("Page", "https://p.test")]))
    events = []
    answer, _ = run("who is the president of france?", on_tool=events.append)
    assert answer == "cached answer here"
    assert events == []


def test_cache_lookup_runs_alongside_the_graph(wire):
    wire.cache = FakeCache(delay=0.3)

    class SlowGraph(FakeGraph):
        async def astream(self, graph_input, config, stream_mode):
            await asyncio.sleep(0.3)  # graph work before its first output (classifier, search)
            async for event in super().astream(graph_input, config, stream_mode):
                yield event

    wire.graph = SlowGraph([passages(1), token("5 t/ha")])
    started = time.perf_counter()
    run()
    assert time.perf_counter() - started < 0.5  # ~0.3s in parallel, not 0.6s one after the other
    assert wire.cache.saved == ["5 t/ha"]


def test_small_talk_skips_the_cache_lookup(wire):
    wire.cache = FakeCache(hit=HIT)
    wire.graph = FakeGraph([token("Hi!", node="converse")])
    answer, _ = run("hey")
    assert answer == "Hi!"
    assert wire.cache.probes == 0


def test_generation_failure_becomes_rag_error(wire):
    wire.graph = FakeGraph([], error=RuntimeError("llm down"))
    with pytest.raises(RAGError):
        run()


def test_search_failure_keeps_its_own_error_type(wire):
    wire.graph = FakeGraph([], error=DocumentRetrievalError("weaviate down"))
    with pytest.raises(DocumentRetrievalError):
        run()


def web_turn(results):
    from langchain_core.documents import Document

    web_passages = [
        (Document(page_content="text", metadata={"file_name": title, "url": url}), 0.9) for title, url in results
    ]
    return [
        ("updates", {"transform_query": {"retrieval_query": None, "web_query": "president of France"}}),
        ("updates", {"web_search": {"web_passages": web_passages}}),
        token("Macron", node="web_answer"),
    ]


def test_web_search_is_reported_as_a_tool_call_with_its_sources(wire):
    wire.graph = FakeGraph(web_turn([("Élysée", "https://elysee.fr")]))
    events = []
    answer, _ = run("who is the president of france?", on_tool=events.append)
    assert answer == "Macron"
    assert [(e.name, e.status, e.input) for e in events] == [
        ("web_search", "running", "president of France"),
        ("web_search", "done", "president of France"),
    ]
    assert [(s.title, s.url) for s in events[1].sources] == [("Élysée", "https://elysee.fr")]


def test_web_search_without_results_is_reported_as_failed(wire):
    wire.graph = FakeGraph(web_turn([])[:2] + [token("Hi", node="converse")])
    events = []
    run("who is the president of france?", on_tool=events.append)
    assert [e.status for e in events] == ["running", "failed"]


def test_document_questions_report_no_tool_calls(wire):
    wire.graph = FakeGraph([("updates", {"transform_query": {"retrieval_query": "yield", "web_query": None}}), token("5 t/ha")])
    events = []
    run(on_tool=events.append)
    assert events == []
