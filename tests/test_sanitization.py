import pytest

from app.vector_db.sanitization import sanitize_index_name, sanitize_namespace


@pytest.mark.parametrize(
    "raw, expected",
    [("voice-chat", "VoiceChat"), ("docs_2024", "Docs2024"), ("2024docs", "C2024docs")],
)
def test_index_name_becomes_pascal_case_starting_with_a_letter(raw, expected):
    assert sanitize_index_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "---"])
def test_index_name_without_usable_characters_is_rejected(raw):
    with pytest.raises(ValueError):
        sanitize_index_name(raw)


def test_namespace_replaces_unsafe_characters_and_keeps_safe_ones():
    assert sanitize_namespace(" team a/b_c-d ") == "team-a-b_c-d"


def test_namespace_is_cut_to_64_characters():
    assert len(sanitize_namespace("x" * 100)) == 64


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_namespace_is_rejected(raw):
    with pytest.raises(ValueError):
        sanitize_namespace(raw)
