import pytest

from ai_workflow.retrieval.question_rules import is_small_talk, is_standalone, normalize_question


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


@pytest.mark.parametrize(
    "message", ["hey", "Hi there!", "hello", "Good morning", "Thanks!", "thank you so much", "ok", "Got it.",
                "bye", "How are you?", "What’s up?"],
)
def test_bare_small_talk_is_recognised(message):
    assert is_small_talk(message)


@pytest.mark.parametrize(
    "message", ["hey, what is the leave policy?", "thanks, and what about sick leave?", "hi what can you do",
                "what can you do?", "ok so who approved the policy", "How are the sales in FY2026?", "great wall of china"],
)
def test_messages_with_a_request_are_not_small_talk(message):
    assert not is_small_talk(message)
