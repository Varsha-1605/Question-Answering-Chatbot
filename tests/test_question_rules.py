import pytest

from ai_workflow.retrieval.question_rules import is_standalone, normalize_question


def test_normalize_ignores_case_spacing_and_trailing_punctuation():
    assert normalize_question("  What   is RAG?! ") == "what is rag"


def test_normalize_folds_unicode_width():
    assert normalize_question("ＲＡＧ？") == "rag"


def test_without_history_everything_is_standalone():
    assert is_standalone("and what about it?", has_history=False)


@pytest.mark.parametrize("question", ["and what about it?", "Explain that again", "how about the former"])
def test_back_reference_needs_history(question):
    assert not is_standalone(question, has_history=True)


def test_self_contained_question_is_standalone_even_with_history():
    assert is_standalone("What is the yield of Varuna rice?", has_history=True)
