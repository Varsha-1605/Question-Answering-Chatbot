from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "varuna_rice_trial_report.pdf"

pytestmark = pytest.mark.integration


def test_ingest_a_pdf_then_get_an_answer_about_it():
    client = TestClient(main.create_app())

    (result,) = client.post("/ingest", files=[("files", (SAMPLE.name, SAMPLE.read_bytes(), "application/pdf"))]).json()
    assert result["status"] == "COMPLETED", result

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"question": "What is this trial report about?"})
        answer = ""
        while (frame := ws.receive_json())["type"] != "end":
            assert frame["type"] == "stream", frame
            answer += frame["message"]
    assert answer.strip()
