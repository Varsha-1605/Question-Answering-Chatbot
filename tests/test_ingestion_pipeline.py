import asyncio

import pytest
from langchain_core.documents import Document

from ai_workflow.injestion import utils
from ai_workflow.injestion.models import IngestRequestDto, RequestStatusEnum
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.models import PushToDatabaseResponseDto


@pytest.fixture
def steps(monkeypatch):
    """Fake collaborators; ``steps.calls`` records call order."""

    class Steps:
        calls = []
        chunks = [Document(page_content="chunk")]
        push = PushToDatabaseResponseDto(status=True, chunks_stored=1)
        push_error = None
        invalidated = []

    async def validate_and_create_index(index_name):
        Steps.calls.append("index")

    def file_loader(**kwargs):
        Steps.calls.append("parse")
        return Steps.chunks

    async def push_to_database(texts, index_name, namespace, drop_status):
        Steps.calls.append("store")
        if Steps.push_error:
            raise Steps.push_error
        Steps.pushed = (index_name, namespace, drop_status)
        return Steps.push

    async def invalidate(index_name, namespace):
        Steps.invalidated.append((index_name, namespace))

    Steps.calls, Steps.invalidated, Steps.push_error = [], [], None
    monkeypatch.setattr(utils, "validate_and_create_index", validate_and_create_index)
    monkeypatch.setattr(utils, "file_loader", file_loader)
    monkeypatch.setattr(utils, "push_to_database", push_to_database)
    monkeypatch.setattr(utils, "invalidate_cached_answers", invalidate)
    return Steps


def ingest(**overrides):
    request = IngestRequestDto(
        pre_signed_url="/tmp/a.txt", file_name="a.txt", original_file_name="a.txt", file_type="txt", **overrides
    )
    return asyncio.run(utils.load_file_and_push_to_database(request))


def test_runs_index_then_parse_then_store(steps):
    result = ingest()
    assert steps.calls == ["index", "parse", "store"]
    assert result.status == RequestStatusEnum.COMPLETED and result.total_pages == 1


def test_missing_index_and_namespace_fall_back_to_the_configured_ones(steps):
    ingest()
    assert steps.pushed[:2] == (config_settings.INDEX_NAME, config_settings.DEFAULT_NAMESPACE)


def test_new_content_invalidates_cached_answers(steps):
    ingest(index_name="Idx", namespace="ns")
    assert steps.invalidated == [("Idx", "ns")]


def test_unchanged_file_does_not_invalidate_cached_answers(steps):
    steps.push = PushToDatabaseResponseDto(status=True, chunks_stored=0)
    result = ingest()
    assert result.status == RequestStatusEnum.COMPLETED
    assert steps.invalidated == []


def test_file_without_extracted_text_fails_before_storing(steps):
    steps.chunks = []
    result = ingest()
    assert result.status == RequestStatusEnum.FAILED
    assert "store" not in steps.calls


def test_store_rejection_fails_and_keeps_cached_answers(steps):
    steps.push = PushToDatabaseResponseDto(status=False, error="weaviate said no")
    result = ingest()
    assert (result.status, result.error_detail) == (RequestStatusEnum.FAILED, "weaviate said no")
    assert steps.invalidated == []


def test_unexpected_error_becomes_a_failed_result_not_an_exception(steps):
    steps.push_error = RuntimeError("boom")
    result = ingest()
    assert (result.status, result.error_detail) == (RequestStatusEnum.FAILED, "boom")
