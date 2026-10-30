import asyncio

from ai_workflow.retrieval.web_search import WebSearch


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def make_tool(client):
    tool = WebSearch(api_key="test", max_results=3, timeout=5)
    tool.client = client
    return tool


def test_results_become_passages_with_title_and_url():
    client = FakeClient({"results": [{"title": "Page", "url": "https://x.test", "content": "text", "score": 0.9}]})
    passages = asyncio.run(make_tool(client).search("news"))
    document, score = passages[0]
    assert document.page_content == "text"
    assert document.metadata == {"file_name": "Page", "url": "https://x.test"}
    assert score == 0.9
    assert client.calls == [{"query": "news", "max_results": 3, "timeout": 5}]


def test_results_without_content_are_dropped():
    client = FakeClient({"results": [{"title": "Empty", "url": "https://x.test", "content": ""}]})
    assert asyncio.run(make_tool(client).search("news")) == []


def test_a_failing_search_returns_no_passages():
    assert asyncio.run(make_tool(FakeClient(error=RuntimeError("down"))).search("news")) == []
