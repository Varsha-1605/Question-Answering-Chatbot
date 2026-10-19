import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pymupdf
from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.injestion.multimodal.elements import ExtractionLimits

# Formats vision models accept as-is; anything else is converted to PNG.
PASS_THROUGH_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
CAPTION_PREFIXES = ("figure", "fig.", "fig ")
CONTEXT_CHARS = 300  # nearby text given to the vision model, each side
RENDER_DPI = 150  # enough to read chart labels; higher only costs tokens
SVG_RENDER_DPI = 96


@dataclass(frozen=True, slots=True)
class PreparedImage:
    data: bytes
    mime_type: str
    width: int
    height: int


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ImageBudget:
    """Per-document image bookkeeping: skips duplicates (logos on every page)
    and stops accepting once the per-file cap is reached."""

    def __init__(self, max_images: int) -> None:
        self._max_images = max_images
        self._seen: set[str] = set()
        self.accepted = 0
        self.skipped = 0

    def is_full(self) -> bool:
        return self.accepted >= self._max_images

    def should_skip(self, digest: str) -> bool:
        """True (and counted as skipped) for a repeat image or when the cap is reached."""
        if digest in self._seen or self.is_full():
            self.skipped += 1
            return True
        return False

    def add(self, digest: str) -> None:
        self._seen.add(digest)
        self.accepted += 1

    def reject(self, digest: Optional[str] = None) -> None:
        """Count an unusable image; remembering its hash skips re-decoding repeats."""
        self.skipped += 1
        if digest:
            self._seen.add(digest)


def prepare_image(raw: bytes, extension: str, limits: ExtractionLimits) -> Optional[PreparedImage]:
    """Decode an embedded image; None if it is unreadable or smaller than the minimum size."""
    extension = extension.lower().lstrip(".")
    if extension == "svg":
        return _render_svg(raw, limits)

    try:
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if min(width, height) < limits.min_image_px:
                return None
            mime_type = PASS_THROUGH_MIME.get(extension)
            if mime_type and max(width, height) <= limits.max_image_side and image.mode in ("RGB", "L", "RGBA"):
                return PreparedImage(raw, mime_type, width, height)
            return _to_png(image, limits.max_image_side)
    except (UnidentifiedImageError, OSError) as error:
        # EMF/WMF and corrupt images end up here; they are counted as skipped.
        logger.debug(f"Skipping unreadable image ({extension}): {error}")
        return None


def render_region(page: pymupdf.Page, rect: pymupdf.Rect, limits: ExtractionLimits) -> Optional[PreparedImage]:
    """Render part of a PDF page (a vector figure or a whole scanned page) to PNG."""
    if min(rect.width, rect.height) <= 0:
        return None
    # Pick the DPI so the longest side fits max_image_side: no resize pass needed.
    dpi = int(min(RENDER_DPI, limits.max_image_side * 72 / max(rect.width, rect.height)))  # PyMuPDF needs an int
    pixmap = page.get_pixmap(clip=rect, dpi=dpi)
    if min(pixmap.width, pixmap.height) < limits.min_image_px:
        return None
    return PreparedImage(pixmap.tobytes("png"), "image/png", pixmap.width, pixmap.height)


def _render_svg(raw: bytes, limits: ExtractionLimits) -> Optional[PreparedImage]:
    try:
        with pymupdf.open(stream=raw, filetype="svg") as svg:
            pixmap = svg[0].get_pixmap(dpi=SVG_RENDER_DPI)
    except (RuntimeError, ValueError) as error:
        logger.debug(f"Skipping unreadable SVG: {error}")
        return None
    if min(pixmap.width, pixmap.height) < limits.min_image_px:
        return None
    with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
        return _to_png(image, limits.max_image_side)


def _to_png(image: Image.Image, max_side: int) -> PreparedImage:
    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    image.thumbnail((max_side, max_side))  # keeps aspect ratio; no-op when already small
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return PreparedImage(buffer.getvalue(), "image/png", image.width, image.height)


def nearby_text(texts: list[str]) -> tuple[list[str], list[str]]:
    """For each position in an ordered list, the text just before and just after it.

    ``texts`` holds "" for non-text items. Two linear passes, each keeping at
    most CONTEXT_CHARS, so the cost stays O(n) for any document size.
    """
    before, recent = [], ""
    for text in texts:
        before.append(recent)
        if text:
            recent = f"{recent} {text}".strip()[-CONTEXT_CHARS:]

    after, upcoming = [], ""
    for text in reversed(texts):
        after.append(upcoming)
        if text:
            upcoming = f"{text} {upcoming}".strip()[:CONTEXT_CHARS]
    after.reverse()
    return before, after
