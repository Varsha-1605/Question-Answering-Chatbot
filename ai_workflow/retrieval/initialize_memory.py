from typing import Iterable, Mapping, Union

from ai_workflow.retrieval.models import Message
from ai_workflow.settings import config_settings

_ROLE_LABELS = {"human": "User", "ai": "Assistant"}
_MAX_CHARS_PER_MESSAGE = 2000


def format_chat_history(
    chat_context: Iterable[Union[Message, Mapping]] | None,
    window: int | None = None,
) -> str:
    """Format the last ``window`` messages as ``User:`` / ``Assistant:`` lines."""
    window = config_settings.CHAT_HISTORY_WINDOW if window is None else window
    if not chat_context or window == 0:
        return ""

    messages = [m if isinstance(m, Message) else Message(**m) for m in chat_context]
    lines = [
        f"{_ROLE_LABELS[m.type]}: {m.content[:_MAX_CHARS_PER_MESSAGE]}"
        for m in messages[-window:]
    ]
    return "\n".join(lines)
