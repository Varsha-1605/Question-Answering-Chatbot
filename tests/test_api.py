import os

import pytest
from fastapi.testclient import TestClient

import main
from app.exception import DocumentRetrievalError, RAGError
from app.injestion import routes as ingest_routes
from app.injestion.models import FileIngestionResponseDto, RequestStatusEnum
from app.retrieval import routes_with_graph as chat
from app.retrieval.rag_util import ConnectionManager
from app.settings import config_settings
from test_run_rag import FakeGraph, token, passages


@pytest.fixture
def client():
    return TestClient(main.create_app())  # no `with`: the lifespan (Postgres, warm-up) is not started


# ---------------------------------------------------------------- health
def test_health_is_always_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("weaviate_up, code", [(True, 200), (False, 503)])
def test_ready_follows_weaviate(client, monkeypatch, weaviate_up, code):
    async def ready():
        return weaviate_up

    monkeypatch.setattr(main, "is_vector_db_ready", ready)
    assert client.get("/ready").status_code == code


def test_ready_is_503_while_warming_up(client):
    client.app.state.warmed_up = False
    response = client.get("/ready")
    assert (response.status_code, response.json()["status"]) == (503, "warming_up")


# ---------------------------------------------------------------- /ingest
@pytest.fixture
def ingested(monkeypatch):
    requests = []

    async def fake_ingest(request):
        requests.append(request)
        assert os.path.exists(request.pre_signed_url)
        return FileIngestionResponseDto(file_name=request.file_name, status=RequestStatusEnum.COMPLETED, total_pages=1)

    monkeypatch.setattr(ingest_routes, "load_file_and_push_to_database", fake_ingest)
    return requests


def upload(name, content=b"hello"):
    return ("files", (name, content, "application/octet-stream"))


def test_each_uploaded_file_gets_its_own_result(client, ingested):
    response = client.post("/ingest", files=[upload("a.txt"), upload("b.csv")])
    assert response.status_code == 200
    assert sorted(r["file_name"] for r in response.json()) == ["a.txt", "b.csv"]


def test_target_fields_from_the_client_are_ignored(client, ingested):
    client.post(
        "/ingest",
        files=[upload("a.txt")],
        data={"index_name": "Evil", "namespace": "other", "delete_status": "true"},
    )
    (request,) = ingested
    assert request.index_name == config_settings.INDEX_NAME
    assert request.namespace == config_settings.DEFAULT_NAMESPACE
    assert request.delete_status is False


def test_unsupported_file_type_is_rejected_before_any_work(client, ingested):
    response = client.post("/ingest", files=[upload("a.txt"), upload("virus.exe")])
    assert response.status_code == 400
    assert ingested == []


def test_oversized_file_fails_without_being_ingested(client, ingested, monkeypatch):
    monkeypatch.setattr(config_settings, "MAX_FILE_SIZE_MB", 0)
    (result,) = client.post("/ingest", files=[upload("a.txt")]).json()
    assert result["status"] == "FAILED" and "limit" in result["error_detail"]
    assert ingested == []


def test_temp_file_is_removed_afterwards(client, ingested):
    client.post("/ingest", files=[upload("a.txt")])
    assert not os.path.exists(ingested[0].pre_signed_url)


# ---------------------------------------------------------------- WS /ws/chat
@pytest.fixture
def chat_backend(monkeypatch):
    class Backend:
        graph = FakeGraph([passages(1), token("Hi "), token("there")])

    monkeypatch.setattr(chat, "manager", ConnectionManager(max_connections=2))
    monkeypatch.setattr(chat, "get_answer_cache", lambda: None)
    monkeypatch.setattr(chat, "chat_memory_enabled", lambda: False)
    monkeypatch.setattr(chat, "get_graph", lambda: Backend.graph)
    return Backend


def test_ping_gets_pong(client, chat_backend):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "pong"


def test_question_streams_tokens_then_end(client, chat_backend):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"question": "What is the yield?"})
        frames = [ws.receive_json() for _ in range(3)]
    assert [(f["type"], f["message"]) for f in frames] == [("stream", "Hi "), ("stream", "there"), ("end", "")]


def test_chat_works_without_ingested_documents(client, chat_backend):
    chat_backend.graph = FakeGraph([token("Hello!")])
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"question": "hi"})
        assert [ws.receive_json()["type"] for _ in range(2)] == ["stream", "end"]


def test_server_at_capacity_refuses_new_sockets(client, chat_backend):
    with client.websocket_connect("/ws/chat"), client.websocket_connect("/ws/chat"):
        with client.websocket_connect("/ws/chat") as third:
            assert third.receive_json()["code"] == "MAX_CONNECTIONS_REACHED"


def test_closed_socket_frees_its_slot(client, chat_backend):
    for _ in range(3):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"


def test_invalid_request_is_reported_and_the_socket_stays_open(client, chat_backend):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("{not json")
        assert ws.receive_json()["code"] == "INVALID_REQUEST"
        ws.send_text("ping")
        assert ws.receive_text() == "pong"


def test_idle_socket_is_closed_with_a_notice(client, chat_backend, monkeypatch):
    monkeypatch.setattr(config_settings, "WEBSOCKET_IDLE_TIMEOUT_SECONDS", 0.05)
    with client.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["code"] == "IDLE_TIMEOUT"


@pytest.mark.parametrize(
    "error, code",
    [(DocumentRetrievalError("x"), "SEARCH_FAILED"), (RuntimeError("x"), "ANSWER_FAILED")],
)
def test_pipeline_failure_is_reported_with_a_code_and_the_socket_survives(client, chat_backend, error, code):
    chat_backend.graph = FakeGraph([], error=error)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"question": "What is the yield?"})
        assert ws.receive_json()["code"] == code
        ws.send_text("ping")
        assert ws.receive_text() == "pong"
