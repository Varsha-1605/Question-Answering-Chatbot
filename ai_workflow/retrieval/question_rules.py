import re
import unicodedata

# Words that depend on earlier messages.
_REFERS_BACK = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|him|her|his|"
    r"above|previous|earlier|former|latter|same)\b"
    r"|^(and|also|then|so|what about|how about)\b"
)
_TRAILING_PUNCTUATION = "?!.,;:。？！"
# Whole messages that need no classifier, search or cache: answered by plain conversation.
_SMALL_TALK = re.compile(
    r"(hi|hello|hey|hiya|yo|greetings|good (morning|afternoon|evening))( there)?"
    r"|(thanks|thank you|thx|ty|many thanks)( (a lot|so much|very much))?"
    r"|ok|okay|cool|great|nice|awesome|perfect|got it|i see"
    r"|bye|goodbye|see you|see ya"
    r"|how are you( doing)?|how's it going|what's up"
)


def normalize_question(question: str) -> str:
    """Lowercase, collapse spaces and strip trailing punctuation."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = " ".join(text.split())
    return text.strip(_TRAILING_PUNCTUATION + " ")


def is_small_talk(question: str) -> bool:
    """True for a bare greeting, thanks, acknowledgement or goodbye (whole message only)."""
    text = normalize_question(question).replace("’", "'")
    return bool(_SMALL_TALK.fullmatch(text))


def is_standalone(question: str, has_history: bool) -> bool:
    """True if the question makes sense without chat history."""
    if not has_history:
        return True
    return not _REFERS_BACK.search(normalize_question(question))