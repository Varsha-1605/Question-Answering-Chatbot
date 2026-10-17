import re
import unicodedata

# Words that depend on earlier messages.
_REFERS_BACK = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|him|her|his|"
    r"above|previous|earlier|former|latter|same)\b"
    r"|^(and|also|then|so|what about|how about)\b"
)
_TRAILING_PUNCTUATION = "?!.,;:。？！"


def normalize_question(question: str) -> str:
    """Lowercase, collapse spaces and strip trailing punctuation."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = " ".join(text.split())
    return text.strip(_TRAILING_PUNCTUATION + " ")


def is_standalone(question: str, has_history: bool) -> bool:
    """True if the question makes sense without chat history."""
    if not has_history:
        return True
    return not _REFERS_BACK.search(normalize_question(question))