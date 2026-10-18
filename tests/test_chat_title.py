import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import main
from ai_workflow.retrieval import routes_with_graph as chat
from ai_workflow.retrieval import title


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Greeting", "Greeting"),
        ('"Rice yield report."', "Rice yield report"),
        ("**Leave policy**\nExtra explanation", "Leave policy"),
        ("None", None),
        ("   ", None),
        ("x" * 100, "x" * title.MAX_TITLE_CHARS + "…"),
    ],
)
def test_clean_title(raw, expected):
    assert title.clean_title(raw) == expected


def test_generate_chat_title_uses_the_model_output(monkeypatch):
    monkeypatch.setattr(title, "get_rewrite_model", lambda: FakeListChatModel(responses=['"Greeting."']))
    assert asyncio.run(title.generate_chat_title("hey")) == "Greeting"


def test_generate_chat_title_returns_none_when_the_model_fails(monkeypatch):
    def fail(_):
        raise RuntimeError("model down")

    monkeypatch.setattr(title, "get_rewrite_model", lambda: RunnableLambda(fail))
    assert asyncio.run(title.generate_chat_title("hey")) is None


@pytest.fixture
def client():
    return TestClient(main.create_app())


def test_title_endpoint_returns_the_generated_title(client, monkeypatch):
    async def fake_title(question):
        return f"About {question}"

    monkeypatch.setattr(chat, "generate_chat_title", fake_title)
    assert client.post("/chat/title", json={"question": "rice"}).json() == {"title": "About rice"}


def test_title_endpoint_rejects_an_empty_question(client):
    assert client.post("/chat/title", json={"question": ""}).status_code == 422
