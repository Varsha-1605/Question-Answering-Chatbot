from functools import lru_cache
from urllib.parse import urlparse


from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable
from loguru import logger

from ai_workflow.exception import ConfigurationError
from ai_workflow.settings import LLMServiceType, config_settings
from langchain_litellm import ChatLiteLLMRouter

from ai_workflow.cache.embeddings import build_cached_embeddings

def get_provider_endpoint() -> tuple[str, int]:
    """Return the (host, port) of the LLM provider."""
    service = config_settings.LLM_SERVICE

    if service == LLMServiceType.AWS_BEDROCK:
        return f"bedrock-runtime.{config_settings.AWS_REGION_NAME}.amazonaws.com", 443

    url = (
        config_settings.OLLAMA_API_URL
        if service == LLMServiceType.OLLAMA
        else config_settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
    )
    parsed = urlparse(url)
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def _secret(value) -> str | None:
    return value.get_secret_value() if value else None


def _bedrock_credentials() -> dict:
    """AWS keys if configured; otherwise boto3's default credentials."""
    creds = {
        "aws_access_key_id": config_settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": config_settings.AWS_SECRET_ACCESS_KEY,
        "aws_session_token": config_settings.AWS_SESSION_TOKEN,
    }
    return {key: value for key, value in creds.items() if value}


@lru_cache(maxsize=8)
def get_chat_model(temperature: float | None = None, model: str | None = None) -> BaseChatModel:
    """Return the chat model for ``LLM_SERVICE`` (``model`` overrides the name)."""
    temperature = config_settings.LLM_TEMPERATURE if temperature is None else temperature
    service = config_settings.LLM_SERVICE

    if config_settings.LLM_FALLBACK_ENABLED:
        return ChatLiteLLMRouter(router=get_llm_router(),
                                model_name="answer",
                                model=f"openai/{config_settings.OPENAI_CHAT_MODEL}",
                                temperature=temperature)

    if service == LLMServiceType.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or config_settings.OPENAI_CHAT_MODEL,
            temperature=temperature,
            api_key=_secret(config_settings.OPENAI_API_KEY),
            base_url=config_settings.OPENAI_BASE_URL,
            timeout=config_settings.LLM_TIMEOUT_SECONDS,
            max_retries=config_settings.LLM_MAX_RETRIES,
        )

    if service == LLMServiceType.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=config_settings.OLLAMA_API_URL,
            model=model or config_settings.OLLAMA_CHAT_MODEL,
            temperature=temperature,
        )

    if service == LLMServiceType.AWS_BEDROCK:
        from langchain_aws import ChatBedrock

        model_id = model or config_settings.BEDROCK_CHAT_MODEL_ID
        provider = config_settings.BEDROCK_CHAT_PROVIDER
        if model_id.startswith("arn:") and not provider:
            raise ConfigurationError(
                "BEDROCK_CHAT_PROVIDER is required when BEDROCK_CHAT_MODEL_ID is an ARN"
            )
        logger.info(f"Using Bedrock chat model {model_id}")
        return ChatBedrock(
            model_id=model_id,
            provider=provider,
            region_name=config_settings.AWS_REGION_NAME,
            temperature=temperature,
            **_bedrock_credentials(),
        )

    raise ConfigurationError(f"Unsupported LLM_SERVICE: {service}")


def get_rewrite_model() -> BaseChatModel:
    """Return the query-rewrite model (``REWRITE_MODEL`` or the chat model)."""
    if config_settings.LLM_FALLBACK_ENABLED:
        return ChatLiteLLMRouter(router=get_llm_router(), model_name="rewrite",
                         model=f"openai/{config_settings.REWRITE_MODEL or config_settings.OPENAI_CHAT_MODEL}",
                         temperature=0)
    return get_chat_model(model=config_settings.REWRITE_MODEL or None)


class TracedEmbeddings(Embeddings):
    """Embeddings wrapper that traces each call in LangSmith."""

    def __init__(self, query_model: Embeddings, document_model: Embeddings) -> None:
        self._query = query_model
        self._documents = document_model

    @traceable(
        name="embed_documents",
        run_type="embedding",
        process_inputs=lambda inputs: {"num_texts": len(inputs["texts"])},
        process_outputs=lambda vectors: {
            "num_vectors": len(vectors),
            "dimensions": len(vectors[0]) if vectors else 0,
        },
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._documents.embed_documents(texts)

    @traceable(
        name="embed_query",
        run_type="embedding",
        process_inputs=lambda inputs: {"text": inputs["text"]},
        process_outputs=lambda vector: {"dimensions": len(vector)},
    )
    def embed_query(self, text: str) -> list[float]:
       return self._query.embed_query(text)



def embedding_model_id() -> str:
    """Provider and embedding model name, used in cache keys."""
    service = config_settings.LLM_SERVICE
    model = {
        LLMServiceType.OPENAI: config_settings.OPENAI_EMBEDDING_MODEL,
        LLMServiceType.OLLAMA: config_settings.OLLAMA_EMBEDDING_MODEL,
        LLMServiceType.AWS_BEDROCK: config_settings.BEDROCK_EMBEDDING_MODEL_ID,
    }[service]
    return f"{service.value}:{model}"


def chat_model_id() -> str:
    """Provider and chat model name, used in cache keys."""
    service = config_settings.LLM_SERVICE
    model = {
        LLMServiceType.OPENAI: config_settings.OPENAI_CHAT_MODEL,
        LLMServiceType.OLLAMA: config_settings.OLLAMA_CHAT_MODEL,
        LLMServiceType.AWS_BEDROCK: config_settings.BEDROCK_CHAT_MODEL_ID,
    }[service]
    return f"{service.value}:{model}:t{config_settings.LLM_TEMPERATURE}"



@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    embeddings = TracedEmbeddings(
        query_model=_build_embeddings(
            timeout=config_settings.EMBEDDING_QUERY_TIMEOUT_SECONDS,
            max_retries=config_settings.EMBEDDING_QUERY_MAX_RETRIES,
        ),
        document_model=_build_embeddings(
            timeout=config_settings.LLM_TIMEOUT_SECONDS,
            max_retries=config_settings.LLM_MAX_RETRIES,
        ),
    )
    if not config_settings.CACHE_ENABLED:
        return embeddings
    return build_cached_embeddings(embeddings, model_id=embedding_model_id())


def _build_embeddings(timeout: float, max_retries: int) -> Embeddings:
    service = config_settings.LLM_SERVICE

    if service == LLMServiceType.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config_settings.OPENAI_EMBEDDING_MODEL,
            api_key=_secret(config_settings.OPENAI_API_KEY),
            base_url=config_settings.OPENAI_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    if service == LLMServiceType.OLLAMA:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=config_settings.OLLAMA_API_URL,
            model=config_settings.OLLAMA_EMBEDDING_MODEL,
        )

    if service == LLMServiceType.AWS_BEDROCK:
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            model_id=config_settings.BEDROCK_EMBEDDING_MODEL_ID,
            region_name=config_settings.AWS_REGION_NAME,
            **_bedrock_credentials(),
        )

    raise ConfigurationError(f"Unsupported LLM_SERVICE: {service}")


@lru_cache(maxsize=1)
def get_llm_router():
    """LiteLLM router: OpenAI first, OpenRouter models as fallback.

    Over-budget or failing calls fall back to Llama/Gemini. The rewrite and
    answer models must differ, since OpenAI limits are per model.
    """
    import litellm
    from litellm import Router
    from litellm.types.router import RetryPolicy

    s = config_settings
    overflow = s.LOAD_BALANCING_MODE == "overflow"

    def entry(name: str, model: str, key, rpm: int | None = None, tpm: int | None = None,
              order: int | None = None, priced_as: str | None = None) -> dict:
        params = {"model": model, "api_key": _secret(key)}
        if rpm:
            params["rpm"] = rpm
        if tpm:
            params["tpm"] = tpm
        if order and overflow:
            params["order"] = order
        if priced_as:
            price = litellm.model_cost.get(priced_as, {})
            for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
                if field in price:
                    params[field] = price[field]
        return {"model_name": name, "litellm_params": params}

    shared_counts = {"redis_url": s.REDIS_URL.get_secret_value()} if s.REDIS_URL else {}
    if not shared_counts:
        logger.warning("REDIS_URL not set: OpenAI budgets are counted per process, not shared")

    rewrite_model = s.REWRITE_MODEL or s.OPENAI_CHAT_MODEL
    copies = []
    if s.LOAD_BALANCING_ENABLED:
        copies = [
            entry("answer", f"openrouter/openai/{s.OPENAI_CHAT_MODEL}", s.OPENROUTER_API_KEY,
                  s.OPENROUTER_COPY_RPM_BUDGET, order=2, priced_as=s.OPENAI_CHAT_MODEL),
            entry("rewrite", f"openrouter/openai/{rewrite_model}", s.OPENROUTER_API_KEY,
                  s.OPENROUTER_COPY_RPM_BUDGET, order=2, priced_as=rewrite_model),
        ]
        logger.info(f"LLM load balancing on ({s.LOAD_BALANCING_MODE}): OpenAI + OpenRouter copies")

    return Router(
        model_list=[
            entry("answer", f"openai/{s.OPENAI_CHAT_MODEL}", s.OPENAI_API_KEY,
                  s.ANSWER_MODEL_RPM_BUDGET, s.ANSWER_MODEL_TPM_BUDGET, order=1),
            entry("rewrite", f"openai/{rewrite_model}", s.OPENAI_API_KEY,
                  s.REWRITE_MODEL_RPM_BUDGET, s.REWRITE_MODEL_TPM_BUDGET, order=1),
            *copies,
            entry("llama-answer", "openrouter/meta-llama/llama-3.3-70b-instruct", s.OPENROUTER_API_KEY),
            entry("llama-rewrite", "openrouter/meta-llama/llama-3.1-8b-instruct", s.OPENROUTER_API_KEY),
            entry("gemini", "openrouter/google/gemini-2.5-flash", s.OPENROUTER_API_KEY),
        ],
        fallbacks=[
            {"answer": ["llama-answer", "gemini"]},
            {"rewrite": ["llama-rewrite", "gemini"]},
        ],
        routing_strategy="usage-based-routing-v2",
        # Never retry a used-up budget; fall back instead.
        retry_policy=RetryPolicy(
            RateLimitErrorRetries=0,
            TimeoutErrorRetries=1,
            InternalServerErrorRetries=1,
            ServiceUnavailableErrorRetries=1,
        ),
        timeout=s.FALLBACK_TIMEOUT_SECONDS,
        **shared_counts,
    )