import asyncio

import pytest
from langchain_core.documents import Document

from app.exception import DocumentRetrievalError
from app.retrieval import reranker
from app.retrieval.reranker import RerankingRetriever


def passage(text, score=0.0):
    return Document(page_content=text), score


class FakeCrossEncoder:
    """Scores by passage length."""

    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


@pytest.fixture
def search(monkeypatch):
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return [passage("a", 0.9), passage("bbb", 0.5), passage("cc", 0.1)]

    monkeypatch.setattr(reranker, "get_related_docs_with_score", fake_search)
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeCrossEncoder())
    return calls


def retrieve(retriever):
    return asyncio.run(retriever.retrieve("q", "search q", "Idx", "ns"))


def test_results_are_sorted_by_cross_encoder_and_cut_to_top_k(search):
    result = retrieve(RerankingRetriever(fetch_k=10, top_k=2))
    assert [doc.page_content for doc, _ in result] == ["bbb", "cc"]
    assert search[0]["total_docs_to_retrieve"] == 10
    assert search[0]["question"] == "search q"


def test_candidates_below_min_score_never_reach_the_reranker(search):
    result = retrieve(RerankingRetriever(fetch_k=10, top_k=5, min_score=0.4))
    assert {doc.page_content for doc, _ in result} == {"a", "bbb"}


def test_disabled_reranker_fetches_top_k_and_keeps_search_order(search):
    result = retrieve(RerankingRetriever(fetch_k=10, top_k=2, enabled=False))
    assert search[0]["total_docs_to_retrieve"] == 2
    assert [doc.page_content for doc, _ in result] == ["a", "bbb", "cc"]  # the fake search ignores the limit


def test_model_failure_becomes_document_retrieval_error(search, monkeypatch):
    class Broken:
        def predict(self, pairs):
            raise RuntimeError("boom")

    monkeypatch.setattr(reranker, "get_reranker", lambda: Broken())
    with pytest.raises(DocumentRetrievalError):
        retrieve(RerankingRetriever(fetch_k=10, top_k=2))
