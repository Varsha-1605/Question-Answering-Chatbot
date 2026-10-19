from dataclasses import dataclass, field
from typing import Literal, Optional, Union

BBox = tuple[float, float, float, float]
ImageSource = Literal["embedded", "vector_figure", "scanned_page", "docx_inline"]


@dataclass(frozen=True, slots=True)
class Location:
    order: int  # position in the whole document (0, 1, 2, ...)
    page: Optional[int] = None  # 1-based; None for DOCX (no pages)
    section: str = ""
    section_path: str = ""  # "Heading 1 > Heading 2"
    bbox: Optional[BBox] = None  # PDF only, in points


@dataclass(frozen=True, slots=True)
class TextElement:
    text: str
    location: Location
    heading_level: Optional[int] = None  # DOCX heading depth of the current section


@dataclass(frozen=True, slots=True)
class TableElement:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    location: Location
    title: str = ""
    table_index: int = 0  # 1-based within the document


@dataclass(frozen=True, slots=True)
class ImageElement:
    image_bytes: bytes = field(repr=False)
    mime_type: str
    width: int
    height: int
    image_hash: str  # sha256 of the original bytes; dedup and cache key
    source: ImageSource
    location: Location
    caption: str = ""
    context: str = ""  # nearby document text, given to the vision model
    image_index: int = 0  # 1-based within the document


Element = Union[TextElement, TableElement, ImageElement]


@dataclass(slots=True)
class ExtractedDocument:
    elements: list[Element] = field(default_factory=list)
    total_pages: Optional[int] = None
    images_skipped: int = 0  # too small, duplicate, unreadable or over the cap


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    min_image_px: int
    max_images: int
    max_image_side: int = 1600  # larger images are downscaled before the vision call
