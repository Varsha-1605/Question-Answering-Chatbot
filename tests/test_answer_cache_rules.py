import pytest

from app.retrieval import answer_cache
from app.retrieval.answer_cache import answer_scope, is_cacheable_answer


@pytest.mark.parametrize(
    "answer",
    ["", "   ", "I don't have that specific information.", "I don’t have that specific information"],
)
def test_empty_and_not_found_answers_are_not_cached(answer):
    assert not is_cacheable_answer(answer)


def test_real_answer_is_cached():
    assert is_cacheable_answer("The yield is 5 t/ha.")


@pytest.fixture
def fixed_models(monkeypatch):
    monkeypatch.setattr(answer_cache, "chat_model_id", lambda: "chat-1")
    monkeypatch.setattr(answer_cache, "embedding_model_id", lambda: "embed-1")
    monkeypatch.setattr(answer_cache, "_answer_prompt_fingerprint", lambda: "prompt-1")


def test_scope_is_stable_and_ignores_language_case(fixed_models):
    assert answer_scope("Idx/ns", "v1", "English") == answer_scope("Idx/ns", "v1", " english ")


@pytest.mark.parametrize(
    "other",
    [("Idx/other", "v1", "english"), ("Idx/ns", "v2", "english"), ("Idx/ns", "v1", "hindi")],
)
def test_scope_changes_with_documents_version_or_language(fixed_models, other):
    assert answer_scope("Idx/ns", "v1", "english") != answer_scope(*other)


@pytest.mark.parametrize("name", ["chat_model_id", "embedding_model_id", "_answer_prompt_fingerprint"])
def test_scope_changes_with_model_or_prompt(fixed_models, monkeypatch, name):
    before = answer_scope("Idx/ns", "v1", "english")
    monkeypatch.setattr(answer_cache, name, lambda: "changed")
    assert answer_scope("Idx/ns", "v1", "english") != before
