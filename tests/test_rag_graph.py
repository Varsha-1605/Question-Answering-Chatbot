import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ai_workflow.retrieval import graph


class FakeRetriever:
    searches = []
    fail_on = None  # search_query that raises

    async def retrieve(self, question, search_query, index_name, namespace):
        self.searches.append(search_query)
        if search_query == self.fail_on:
            raise RuntimeError("search down")
        return [(Document(page_content=f"passage for {search_query}", metadata={"file_name": "f.pdf", "page": 2}), 1.0)]


@pytest.fixture
def pipeline(monkeypatch):
    class Pipeline:
        rewrite = None  # what the query rewrite returns; the classifier's label follows it
        retriever = FakeRetriever
        replies = ["the reply"]
        rewrites = 0

    Pipeline.retriever.searches, Pipeline.retriever.fail_on = [], None

    async def rewrite(question, chat_history):
        Pipeline.rewrites += 1
        return Pipeline.rewrite

    async def classify(question, chat_history):
        if Pipeline.rewrite is None:
            return "NONE"
        return "WEB" if Pipeline.rewrite.startswith("WEB:") else "DOC"

    monkeypatch.setattr(graph, "classify_user_query", classify)

    class RetrieverFactory:
        @staticmethod
        def from_settings():
            return Pipeline.retriever()

    monkeypatch.setattr(graph, "transform_user_query_for_retrieval", rewrite)
    monkeypatch.setattr(graph, "RerankingRetriever", RetrieverFactory)
    monkeypatch.setattr(graph, "get_chat_model", lambda: FakeListChatModel(responses=Pipeline.replies))
    return Pipeline


def ask(question, chat_history=""):
    return asyncio.run(
        graph.build_rag_graph().ainvoke(
            {
                "question": question,
                "chat_history": chat_history,
                "language": "English",
                "index_name": "Idx",
                "namespace": "ns",
            }
        )
    )


def test_format_context_numbers_and_labels_passages():
    passages = [
        (Document(page_content="one", metadata={"file_name": "a.pdf", "page": 4}), 1.0),
        (Document(page_content="two", metadata={}), 1.0),
    ]
    assert graph.format_context(passages) == "[1] (a.pdf, page 4)\none\n\n[2] (unknown)\ntwo"


def test_small_talk_is_answered_without_using_any_passages(pipeline):
    pipeline.rewrite = None
    result = ask("hello there")
    assert result["answer"] == "the reply"
    assert "passages" not in result


def test_standalone_question_reuses_the_search_made_with_the_question_as_typed(pipeline):
    pipeline.rewrite = "rewritten query"
    result = ask("What is the yield of Varuna rice?")
    assert pipeline.retriever.searches == ["What is the yield of Varuna rice?"]
    assert result["passages"]


def test_follow_up_is_searched_again_with_the_rewritten_query(pipeline):
    pipeline.rewrite = "yield of Varuna rice in 2023"
    result = ask("and what about that one?", chat_history="Human: yield of Varuna?\nAI: 5 t/ha")
    assert pipeline.retriever.searches[-1] == "yield of Varuna rice in 2023"
    assert "yield of Varuna rice in 2023" in result["passages"][0][0].page_content


@pytest.fixture
def rewrite_all(monkeypatch):
    """Rewrite every question, not only follow-ups."""
    monkeypatch.setattr(graph.config_settings, "REWRITE_STANDALONE_QUESTIONS", True)


def test_failed_raw_search_falls_back_to_the_rewritten_query(pipeline, rewrite_all):
    pipeline.rewrite = "rewritten query"
    pipeline.retriever.fail_on = "What is the yield?"
    result = ask("What is the yield?")
    assert pipeline.retriever.searches[-1] == "rewritten query"
    assert result["answer"] == "the reply"


def test_finished_turn_is_recorded_as_question_and_answer(pipeline):
    result = ask("hello")
    assert [m.content for m in result["messages"]] == ["hello", "the reply"]


WEB_RESULTS = [(Document(page_content="web fact", metadata={"file_name": "News", "url": "https://n.test"}), 0.9)]


class FakeWebSearch:
    queries = []
    results = WEB_RESULTS

    async def search(self, query):
        self.queries.append(query)
        return self.results


@pytest.fixture
def web(monkeypatch, pipeline):
    FakeWebSearch.queries, FakeWebSearch.results = [], WEB_RESULTS

    class Factory:
        @staticmethod
        def from_settings():
            return FakeWebSearch()

    monkeypatch.setattr(graph, "WebSearch", Factory)
    return FakeWebSearch


def test_current_events_question_is_answered_from_the_web(web, pipeline, rewrite_all):
    pipeline.rewrite = "WEB: latest OpenAI news"
    result = ask("What is the latest news about OpenAI?")
    assert web.queries == ["latest OpenAI news"]
    assert result["answer"] == "the reply"
    assert result["web_passages"]
    assert "passages" not in result


def test_web_question_with_no_results_is_answered_by_conversation(web, pipeline):
    pipeline.rewrite = "WEB: latest OpenAI news"
    web.results = []
    result = ask("What is the latest news about OpenAI?")
    assert result["answer"] == "the reply"
    assert "passages" not in result


def test_web_marker_without_a_tavily_key_is_plain_conversation(pipeline, monkeypatch):
    class NoWeb:
        @staticmethod
        def from_settings():
            return None

    monkeypatch.setattr(graph, "WebSearch", NoWeb)
    pipeline.rewrite = "WEB: latest OpenAI news"
    result = ask("What is the latest news about OpenAI?")
    assert result["answer"] == "the reply"
    assert "web_passages" not in result


def test_format_web_context_labels_title_and_url():
    passages = [(Document(page_content="text", metadata={"file_name": "News", "url": "https://n.test"}), 1.0)]
    assert graph.format_web_context(passages) == "[1] (News, https://n.test)\ntext"


def test_standalone_question_is_classified_but_not_rewritten(pipeline):
    pipeline.rewrite = "rewritten query"
    result = ask("What is the yield of Varuna rice?")
    assert pipeline.rewrites == 0
    assert result["retrieval_query"] == "What is the yield of Varuna rice?"


def test_standalone_web_question_searches_the_web_with_the_question_as_typed(web, pipeline):
    pipeline.rewrite = "WEB: anything"
    ask("Who is the president of France?")
    assert web.queries == ["Who is the president of France?"]
    assert pipeline.rewrites == 0


def test_follow_up_is_still_rewritten(pipeline):
    pipeline.rewrite = "Pro plan price"
    result = ask("and how much is it?", chat_history="Human: Tell me about the Pro plan.\nAI: ...")
    assert pipeline.rewrites == 1
    assert result["retrieval_query"] == "Pro plan price"


def test_with_the_flag_on_every_question_is_rewritten(pipeline, rewrite_all):
    pipeline.rewrite = "rewritten query"
    result = ask("What is the yield of Varuna rice?")
    assert pipeline.rewrites == 1
    assert result["retrieval_query"] == "rewritten query"


def test_small_talk_skips_the_classifier_and_the_search(pipeline):
    pipeline.rewrite = "rewritten query"  # would be a document question if the classifier ran
    result = ask("Thanks!")
    assert pipeline.rewrites == 0
    assert pipeline.retriever.searches == []
    assert result["answer"] == "the reply"
    assert "passages" not in result
