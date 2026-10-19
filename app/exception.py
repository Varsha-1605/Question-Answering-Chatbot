class RAGBaseError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RAGBaseError):
    """Missing or invalid configuration."""


class VectorStoreError(RAGBaseError):
    """The vector database rejected or failed an operation."""


class DocumentRetrievalError(RAGBaseError):
    """Retrieving context for a question failed."""


class RAGError(RAGBaseError):
    """The answer-generation pipeline failed."""
