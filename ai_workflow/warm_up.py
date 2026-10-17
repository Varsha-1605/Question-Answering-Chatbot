import asyncio
import socket
import ssl
import time
import uuid
from urllib.parse import urlparse

from loguru import logger

from ai_workflow.cache.client import get_redis_client
from ai_workflow.retrieval.chat_memory import chat_memory_enabled, load_history
from ai_workflow.retrieval.reranker import get_reranker
from ai_workflow.settings import WeaviateConnectionType, config_settings
from ai_workflow.utils import (
    get_chat_model,
    get_embeddings,
    get_provider_endpoint,
    get_rewrite_model,
)
from ai_workflow.vector_db.utils import is_vector_db_ready


def probe_network(host: str, port: int) -> str:
    """Time DNS, TCP and (on 443) TLS to ``host`` (blocking)."""
    timings = {}

    start = time.perf_counter()
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    timings["dns"] = time.perf_counter() - start

    _family, _type, _proto, _name, address = addresses[0]
    start = time.perf_counter()
    connection = socket.create_connection(address[:2], timeout=10)
    timings["tcp"] = time.perf_counter() - start
    try:
        if port == 443:
            start = time.perf_counter()
            context = ssl.create_default_context()
            with context.wrap_socket(connection, server_hostname=host):
                timings["tls"] = time.perf_counter() - start
    finally:
        connection.close()

    return " ".join(f"{name} {seconds:.2f}s" for name, seconds in timings.items())


def _weaviate_endpoint() -> tuple[str, int]:
    if config_settings.WEAVIATE_CONNECTION_TYPE == WeaviateConnectionType.CLOUD:
        parsed = urlparse(config_settings.WEAVIATE_CLUSTER_URL)
        return parsed.hostname, parsed.port or 443
    return config_settings.WEAVIATE_HOST, config_settings.WEAVIATE_HTTP_PORT


async def warm_up() -> None:
    """Run cold-start steps in parallel (network, models, Redis, Postgres, reranker); failures are only logged."""

    async def timed(name: str, step) -> None:
        start = time.perf_counter()
        try:
            detail = await step()
            note = f"  [{detail}]" if isinstance(detail, str) else ""
            logger.info(f"Warm-up {name}: {time.perf_counter() - start:.2f}s{note}")
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Warm-up {name} failed after {time.perf_counter() - start:.2f}s: {error}")

    async def warm_chat_model() -> None:
        model = await asyncio.to_thread(get_chat_model)
        await model.ainvoke("Reply with the single word: ok")

    async def warm_rewrite_model() -> None:
        model = await asyncio.to_thread(get_rewrite_model)
        await model.ainvoke("Reply with the single word: ok")

    async def warm_embedding_model() -> None:
        model = await asyncio.to_thread(get_embeddings)
        # Bypass the cache so the provider connection is warmed.
        provider_model = getattr(model, "underlying_embeddings", model)
        await asyncio.to_thread(provider_model.embed_query, "warm up")

    async def warm_cache() -> None:
        await asyncio.to_thread(lambda: get_redis_client().ping())

    async def warm_chat_memory() -> None:
        await load_history(str(uuid.uuid4()))

    async def warm_reranker() -> None:
        model = await asyncio.to_thread(get_reranker)
        await asyncio.to_thread(model.predict, [("warm up", "warm up")])

    provider_host, provider_port = get_provider_endpoint()
    weaviate_host, weaviate_port = _weaviate_endpoint()

    start = time.perf_counter()
    await asyncio.gather(
        timed(
            f"network {provider_host}",
            lambda: asyncio.to_thread(probe_network, provider_host, provider_port),
        ),
        timed(
            f"network {weaviate_host}",
            lambda: asyncio.to_thread(probe_network, weaviate_host, weaviate_port),
        ),
        timed("weaviate connection", is_vector_db_ready),
        timed("chat model", warm_chat_model),
        *([timed("rewrite model", warm_rewrite_model)] if config_settings.REWRITE_MODEL else []),
        timed("embedding model", warm_embedding_model),
        *([timed("redis cache", warm_cache)] if config_settings.CACHE_ENABLED else []),
        *([timed("postgres chat memory", warm_chat_memory)] if chat_memory_enabled() else []),
        *([timed("reranker", warm_reranker)] if config_settings.RERANKER_ENABLED else []),
    )
    logger.info(f"Warm-up finished in {time.perf_counter() - start:.2f}s")
