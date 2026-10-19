import asyncio
import base64
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional
from weakref import WeakKeyDictionary

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.stores import ByteStore
from loguru import logger

from app.cache.image_descriptions import build_image_description_store
from app.injestion.multimodal.elements import ImageElement
from app.injestion.multimodal.prompts import (
    CONTENT_TYPES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.settings import LLMServiceType, config_settings

DECORATIVE = "decorative"
UNKNOWN_CONTENT_TYPE = "other"
CONTENT_TYPE_PATTERN = re.compile(r"CONTENT TYPE:\s*([a-z]+)", re.IGNORECASE)
# Keeps one figure chunk a sensible size; real descriptions are far shorter.
MAX_DESCRIPTION_CHARS = 8000


@dataclass(frozen=True, slots=True)
class ImageDescription:
    content_type: str
    text: str

    @property
    def is_decorative(self) -> bool:
        return self.content_type == DECORATIVE


@dataclass(slots=True)
class VisionResult:
    descriptions: dict[str, ImageDescription]  # image_hash -> description (failed images are absent)
    failed: int = 0


class ImageDescriber:
    def __init__(self, model: BaseChatModel, model_id: str, max_concurrency: int, store: Optional[ByteStore]) -> None:
        self._model = model
        self._model_id = model_id
        self._max_concurrency = max_concurrency
        self._store = store
        self._semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()

    async def describe(self, images: list[ImageElement]) -> VisionResult:
        unique = {image.image_hash: image for image in images}
        if not unique:
            return VisionResult(descriptions={})

        descriptions = await self._read_cache(list(unique))
        missing = [image for digest, image in unique.items() if digest not in descriptions]
        fresh = await asyncio.gather(*(self._describe_one(image) for image in missing))

        new_entries = {image.image_hash: result for image, result in zip(missing, fresh, strict=True) if result is not None}
        descriptions.update(new_entries)
        await self._write_cache(new_entries)

        failed = len(missing) - len(new_entries)
        logger.info(
            f"Vision: {len(unique)} images, {len(unique) - len(missing)} from cache, "
            f"{len(new_entries)} described, {failed} failed"
        )
        return VisionResult(descriptions=descriptions, failed=failed)

    async def _describe_one(self, image: ImageElement) -> Optional[ImageDescription]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": build_user_prompt(image.caption, image.context)},
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
            ]),
        ]
        async with self._semaphore():
            try:
                response = await self._model.ainvoke(messages)
            except Exception as error:  # noqa: BLE001 - provider errors vary; one image must not fail the file
                logger.warning(
                    f"Vision call failed for image {image.image_index} (page {image.location.page}): "
                    f"{type(error).__name__}"
                )
                return None
        return parse_description(_response_text(response.content))

    def _semaphore(self) -> asyncio.Semaphore:
        # A semaphore belongs to one event loop; keep one per loop so the shared
        # describer also works when tests or scripts start new loops.
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = self._semaphores[loop] = asyncio.Semaphore(self._max_concurrency)
        return semaphore

    # ---- cache ----

    def _cache_key(self, digest: str) -> str:
        # Model and prompt version are part of the key: changing either re-describes.
        return f"{self._model_id}:{PROMPT_VERSION}:{digest}"

    async def _read_cache(self, digests: list[str]) -> dict[str, ImageDescription]:
        if self._store is None:
            return {}
        values = await asyncio.to_thread(self._store.mget, [self._cache_key(d) for d in digests])
        found = {}
        for digest, value in zip(digests, values, strict=True):
            if value is not None:
                found[digest] = ImageDescription(**json.loads(value))
        return found

    async def _write_cache(self, entries: dict[str, ImageDescription]) -> None:
        if self._store is None or not entries:
            return
        pairs = [(self._cache_key(d), json.dumps(asdict(desc)).encode()) for d, desc in entries.items()]
        await asyncio.to_thread(self._store.mset, pairs)


def parse_description(text: str) -> Optional[ImageDescription]:
    """Read the CONTENT TYPE line; None for an empty answer (counted as a failure)."""
    text = text.strip()
    if not text:
        return None
    match = CONTENT_TYPE_PATTERN.search(text.split("\n", 1)[0])
    content_type = match.group(1).lower() if match else UNKNOWN_CONTENT_TYPE
    if content_type not in CONTENT_TYPES:
        content_type = UNKNOWN_CONTENT_TYPE
    return ImageDescription(content_type=content_type, text=text[:MAX_DESCRIPTION_CHARS])


def vision_model_id() -> str:
    """Name of the vision model, stored in chunk metadata and cache keys."""
    if config_settings.LLM_SERVICE == LLMServiceType.OLLAMA:
        return f"ollama:{config_settings.OLLAMA_VISION_MODEL}"
    return f"openai:{config_settings.openai_vision_model}"


@lru_cache(maxsize=1)
def get_image_describer() -> ImageDescriber:
    """One shared describer, so the concurrency limit covers all uploads together."""
    from app.utils import get_vision_model

    return ImageDescriber(
        model=get_vision_model(),
        model_id=vision_model_id(),
        max_concurrency=config_settings.VISION_MAX_CONCURRENCY,
        store=build_image_description_store(),
    )


def _data_url(image: ImageElement) -> str:
    return f"data:{image.mime_type};base64,{base64.b64encode(image.image_bytes).decode('ascii')}"


def _response_text(content) -> str:
    """Chat models return a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
