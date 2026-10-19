import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.retrieval import graph


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
        rewrite = None  # what the query rewrite returns
        retriever = FakeRetriever
        replies = ["the reply"]

    Pipeline.retriever.searches, Pipeline.retriever.fail_on = [], None

    async def rewrite(question, chat_history):
        return Pipeline.rewrite

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


def test_failed_raw_search_falls_back_to_the_rewritten_query(pipeline):
    pipeline.rewrite = "rewritten query"
    pipeline.retriever.fail_on = "What is the yield?"
    result = ask("What is the yield?")
    assert pipeline.retriever.searches[-1] == "rewritten query"
    assert result["answer"] == "the reply"


def test_finished_turn_is_recorded_as_question_and_answer(pipeline):
    result = ask("hello")
    assert [m.content for m in result["messages"]] == ["hello", "the reply"]
