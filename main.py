import os
import warnings
warnings.simplefilter("ignore")
warnings.showwarning = lambda *_args, **_kwargs: None
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.cache.client import close_redis_client
from app.cache.monitoring import log_cache_stats_periodically
from app.exception import RAGBaseError
from app.injestion.models import FILE_TYPE
from app.injestion.routes import router as ingestion_router
from app.retrieval.chat_memory import chat_memory_enabled, close_chat_memory, open_chat_memory
from app.retrieval.routes_with_graph import router as chat_router
from app.settings import config_settings
from app.vector_db.utils import close_vector_db, is_vector_db_ready
from app.warm_up import warm_up

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=config_settings.LOG_LEVEL.upper(),
        serialize=config_settings.ENVIRONMENT == "production",
    )


def start_background_task(enabled: bool, coroutine_factory) -> asyncio.Task | None:
    """Start a background task only when it is enabled."""
    return asyncio.create_task(coroutine_factory()) if enabled else None


def stop_background_task(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    logger.info(
        f"Starting (env={config_settings.ENVIRONMENT}, llm={config_settings.LLM_SERVICE.value}, "
        f"cache={'on' if config_settings.CACHE_ENABLED else 'off'})"
    )

    # Fails startup if POSTGRES_URL is set but unreachable.
    await open_chat_memory()

    # /ready returns 503 until warm-up finishes.
    _app.state.warmed_up = not config_settings.WARM_UP_ON_STARTUP

    async def warm_up_then_mark_ready() -> None:
        await warm_up()
        _app.state.warmed_up = True

    warm_up_task = start_background_task(config_settings.WARM_UP_ON_STARTUP, warm_up_then_mark_ready)
    cache_stats_task = start_background_task(
        config_settings.CACHE_ENABLED and config_settings.CACHE_STATS_LOG_INTERVAL_SECONDS > 0,
        lambda: log_cache_stats_periodically(config_settings.CACHE_STATS_LOG_INTERVAL_SECONDS),
    )

    yield

    stop_background_task(warm_up_task)
    stop_background_task(cache_stats_task)
    await close_chat_memory()
    close_vector_db()
    close_redis_client()


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise RAG", version="0.1.0", lifespan=lifespan)

    if config_settings.CORS_ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config_settings.CORS_ALLOWED_ORIGINS,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["content-type"],
        )

    @app.exception_handler(RAGBaseError)
    async def domain_error_handler(_request, error: RAGBaseError):
        logger.warning(f"Unhandled domain error: {error}")
        return JSONResponse(status_code=500, content={"detail": "Internal error"})

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready() -> JSONResponse:
        """Readiness: warmed up and Weaviate reachable (Redis is optional, so not checked)."""
        if not getattr(app.state, "warmed_up", True):
            return JSONResponse(status_code=503, content={"status": "warming_up"})
        weaviate_ok = await is_vector_db_ready()
        return JSONResponse(
            status_code=200 if weaviate_ok else 503,
            content={"status": "ready" if weaviate_ok else "degraded", "weaviate": weaviate_ok},
        )

    @app.get("/ui-config", tags=["ui"])
    async def ui_config() -> dict:
        """Public defaults and limits for the chat page."""
        return {
            "index_name": config_settings.INDEX_NAME,
            "namespace": config_settings.DEFAULT_NAMESPACE,
            "file_types": FILE_TYPE,
            "max_file_size_mb": config_settings.MAX_FILE_SIZE_MB,
            "max_question_chars": config_settings.MAX_QUESTION_CHARS,
            "persistent_chats": chat_memory_enabled(),
        }

    app.include_router(ingestion_router)
    app.include_router(chat_router)

    if (FRONTEND_DIR / "index.html").is_file():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def chat_page() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()