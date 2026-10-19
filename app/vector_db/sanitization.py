import re

_TENANT_MAX_LEN = 64


def sanitize_index_name(name: str) -> str:
    """Convert to a PascalCase collection name, e.g. ``"voice-chat"`` -> ``"VoiceChat"``."""
    if not name or not name.strip():
        raise ValueError("index name must not be empty")

    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    sanitized = "".join(part[0].upper() + part[1:] for part in parts)
    if not sanitized:
        raise ValueError(f"index name {name!r} has no usable characters")
    if not sanitized[0].isalpha():
        sanitized = "C" + sanitized
    return sanitized


def sanitize_namespace(namespace: str) -> str:
    """Keep letters, digits, ``_`` and ``-`` (max 64 chars)."""
    if not namespace or not namespace.strip():
        raise ValueError("namespace must not be empty")

    sanitized = re.sub(r"[^A-Za-z0-9_-]", "-", namespace.strip())[:_TENANT_MAX_LEN]
    if not sanitized:
        raise ValueError(f"namespace {namespace!r} has no usable characters")
    return sanitized
