import os
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMServiceType(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    AWS_BEDROCK = "aws_bedrock"


class WeaviateConnectionType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # runtime
    ENVIRONMENT: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    # Call Weaviate, the chat model and embeddings once at startup.
    WARM_UP_ON_STARTUP: bool = True

    CORS_ALLOWED_ORIGINS: list[str] = Field(default_factory=list)

    # LLM provider
    LLM_SERVICE: LLMServiceType = LLMServiceType.OPENAI
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 2
    EMBEDDING_QUERY_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0)
    EMBEDDING_QUERY_MAX_RETRIES: int = Field(default=2, ge=0)

    # Optional smaller model for query rewriting; empty uses the chat model.
    REWRITE_MODEL: Optional[str] = None
    # false: standalone questions are only classified and searched as typed.
    # true: they are also rewritten. Follow-ups are always rewritten.
    REWRITE_STANDALONE_QUESTIONS: bool = False

    # Fallback
    OPENROUTER_API_KEY: Optional[SecretStr] = None
    LLM_FALLBACK_ENABLED: bool = False
    FALLBACK_TIMEOUT_SECONDS: int = 20
    # Per-minute OpenAI budgets; over budget, calls go to OpenRouter (needs LLM_FALLBACK_ENABLED).
    ANSWER_MODEL_RPM_BUDGET: int = Field(default=4000, gt=0)
    ANSWER_MODEL_TPM_BUDGET: int = Field(default=1_600_000, gt=0)
    REWRITE_MODEL_RPM_BUDGET: int = Field(default=4000, gt=0)
    REWRITE_MODEL_TPM_BUDGET: int = Field(default=1_600_000, gt=0)
    # Load balancing across OpenAI and an OpenRouter copy (needs LLM_FALLBACK_ENABLED).
    # overflow: OpenAI first, then OpenRouter. split: whichever has used less.
    LOAD_BALANCING_ENABLED: bool = False
    LOAD_BALANCING_MODE: Literal["overflow", "split"] = "overflow"
    # Per-minute request budget per OpenRouter copy; empty means none.
    OPENROUTER_COPY_RPM_BUDGET: Optional[int] = Field(default=None, gt=0)

    # web search (Tavily); off unless a key is set
    TAVILY_API_KEY: Optional[SecretStr] = None
    WEB_SEARCH_MAX_RESULTS: int = Field(default=5, gt=0)
    WEB_SEARCH_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0)

    # openai
    OPENAI_API_KEY: Optional[SecretStr] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ollama
    OLLAMA_API_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:14b"
    OLLAMA_EMBEDDING_MODEL: str = "snowflake-arctic-embed2:latest"

    # aws bedrock (leave keys empty to use the IAM role)
    AWS_REGION_NAME: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[SecretStr] = None
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = None
    AWS_SESSION_TOKEN: Optional[SecretStr] = None
    # Model id or inference-profile ARN.
    BEDROCK_CHAT_MODEL_ID: str = "us.meta.llama3-3-70b-instruct-v1:0"
    # Required only when BEDROCK_CHAT_MODEL_ID is an ARN.
    BEDROCK_CHAT_PROVIDER: Optional[str] = None
    BEDROCK_EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"

    # weaviate
    WEAVIATE_CONNECTION_TYPE: WeaviateConnectionType = WeaviateConnectionType.LOCAL
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_HTTP_PORT: int = 8080
    WEAVIATE_GRPC_PORT: int = 50051
    WEAVIATE_CLUSTER_URL: str = ""
    WEAVIATE_API_KEY: Optional[SecretStr] = None
    # 0 = keyword only, 1 = vector only, between = hybrid.
    WEAVIATE_HYBRID_ALPHA: float = Field(default=0.7, ge=0.0, le=1.0)
    WEAVIATE_TEXT_KEY: str = "text"

    # vector store defaults
    INDEX_NAME: str = "Documents"
    DEFAULT_NAMESPACE: str = "default"

    # ingestion
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=200, ge=0)
    MAX_FILE_SIZE_MB: int = Field(default=200, gt=0)

    # retrieval / chat
    RETRIEVAL_TOP_K: int = Field(default=5, gt=0)
    # Rerank RERANK_CANDIDATES results and keep the best RETRIEVAL_TOP_K.
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Hugging Face token (exported to the environment below).
    HF_TOKEN: Optional[SecretStr] = None
    RERANK_CANDIDATES: int = Field(default=20, gt=0)
    # 0 turns the minimum score off.
    MIN_RELEVANCE_SCORE: float = Field(default=0.0, ge=0.0)
    CHAT_HISTORY_WINDOW: int = Field(default=10, ge=0)
    MAX_WEBSOCKET_CONNECTIONS: int = Field(default=1000, gt=0)
    WEBSOCKET_IDLE_TIMEOUT_SECONDS: int = Field(default=600, gt=0)
    MAX_QUESTION_CHARS: int = Field(default=4000, gt=0)
    MAX_CHAT_CONTEXT_MESSAGES: int = Field(default=50, gt=0)

    # chat memory: when set, conversations are saved in Postgres
    POSTGRES_URL: Optional[SecretStr] = None
    POSTGRES_POOL_MAX_SIZE: int = 10

    # cache (Redis)
    CACHE_ENABLED: bool = False
    REDIS_URL: Optional[SecretStr] = None
    REDIS_CONNECT_TIMEOUT_SECONDS: float = Field(default=0.2, gt=0)
    REDIS_SOCKET_TIMEOUT_SECONDS: float = Field(default=0.2, gt=0)
    CACHE_KEY_PREFIX: str = "rag"
    # How long the cache is skipped after a Redis error.
    CACHE_CIRCUIT_OPEN_SECONDS: float = Field(default=30, gt=0)
    CACHE_QUERY_EMBEDDING_TTL_SECONDS: int = Field(default=30 * 24 * 3600, gt=0)
    CACHE_DOCUMENT_EMBEDDING_TTL_SECONDS: int = Field(default=90 * 24 * 3600, gt=0)
    # Seconds between cache stats logs; 0 turns it off.
    CACHE_STATS_LOG_INTERVAL_SECONDS: int = Field(default=300, ge=0)

    # Semantic answer cache (needs CACHE_ENABLED and Redis 8+ or Redis Stack on database 0).
    SEMANTIC_CACHE_ENABLED: bool = False
    # Minimum cosine similarity for a hit; keep it strict.
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.95, ge=0.5, le=1.0)
    SEMANTIC_CACHE_TTL_SECONDS: int = Field(default=24 * 3600, gt=0)
    DOCUMENT_VERSION_TTL_SECONDS: int = Field(default=365 * 24 * 3600, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.LLM_SERVICE == LLMServiceType.OPENAI and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when LLM_SERVICE=openai")
        if (
            self.WEAVIATE_CONNECTION_TYPE == WeaviateConnectionType.CLOUD
            and not self.WEAVIATE_CLUSTER_URL
        ):
            raise ValueError("WEAVIATE_CLUSTER_URL must be set for cloud connections")

        if self.CACHE_ENABLED and not self.REDIS_URL:
            raise ValueError("REDIS_URL must be set when CACHE_ENABLED=true")
        if self.SEMANTIC_CACHE_ENABLED:
            if not self.CACHE_ENABLED:
                raise ValueError("SEMANTIC_CACHE_ENABLED needs CACHE_ENABLED=true")
            database = urlparse(self.REDIS_URL.get_secret_value()).path.strip("/")
            if database not in ("", "0"):
                raise ValueError("The semantic cache needs Redis database 0 (vector indexes only work there)")

        return self
    


@lru_cache
def get_settings() -> Settings:
    return Settings()


config_settings = get_settings()

# huggingface_hub reads the token from the environment only.
if config_settings.HF_TOKEN and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = config_settings.HF_TOKEN.get_secret_value()