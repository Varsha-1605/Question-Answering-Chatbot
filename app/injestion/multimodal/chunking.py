from dataclasses import dataclass
from collections.abc import Iterator
from itertools import groupby
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.injestion.multimodal.elements import (
    Element,
    ExtractedDocument,
    ImageElement,
    TableElement,
    TextElement,
)
from app.injestion.multimodal.metadata import (
    ChunkMetadata,
    figure_label,
    make_chunk_id,
)
from app.injestion.multimodal.prompts import PROMPT_VERSION
from app.injestion.multimodal.vision import VisionResult

EMBED_IMAGE_AND_TEXT = "image+text"
EMBED_TEXT_ONLY = "text"


@dataclass(frozen=True, slots=True)
class FileInfo:
    file_name: str
    file_type: str
    file_hash: str
    extractor: str
    ingested_at: str  # ISO-8601, one value for the whole file
    total_pages: Optional[int] = None


@dataclass(frozen=True, slots=True)
class Chunk:
    document: Document
    image: Optional[ImageElement] = None  # set for image chunks: embedded from pixels + text

    @property
    def chunk_id(self) -> str:
        return self.document.metadata["chunk_id"]


def build_chunks(
    extracted: ExtractedDocument,
    vision: VisionResult,
    file: FileInfo,
    *,
    chunk_size: int,
    chunk_overlap: int,
    vision_model: str,
    embeds_images: bool,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    builder = _ChunkBuilder(file, vision, splitter, chunk_size, vision_model, embeds_images)
    return list(builder.build(extracted.elements))


class _ChunkBuilder:
    def __init__(self, file, vision, splitter, chunk_size, vision_model, embeds_images) -> None:
        self._file = file
        self._vision = vision
        self._splitter = splitter
        self._chunk_size = chunk_size
        self._vision_model = vision_model
        self._embedding_input = EMBED_IMAGE_AND_TEXT if embeds_images else EMBED_TEXT_ONLY
        self._chunk_index = 0

    def build(self, elements: list[Element]) -> Iterator[Chunk]:
        # Group consecutive elements of the same kind, page and section; tables and
        # images stay one element per group because each is chunked on its own.
        def group_key(element: Element):
            location = element.location
            if isinstance(element, TextElement):
                return ("text", location.page, location.section_path)
            return ("single", location.order)

        for (kind, *_), group in groupby(elements, key=group_key):
            group = list(group)
            if kind == "text":
                yield from self._text_chunks(group)
            elif isinstance(group[0], TableElement):
                yield from self._table_chunks(group[0])
            else:
                yield from self._image_chunks(group[0])

    # ---- text ----

    def _text_chunks(self, elements: list[TextElement]) -> Iterator[Chunk]:
        first = elements[0]
        prefix = f"Section: {first.location.section_path}\n" if first.location.section_path else ""
        pieces = self._splitter.split_text("\n".join(element.text for element in elements))
        for part, piece in enumerate(pieces, 1):
            yield self._chunk(
                f"{prefix}{piece}", "text", first, part, len(pieces),
                heading_level=first.heading_level,
            )

    # ---- tables ----

    def _table_chunks(self, table: TableElement) -> Iterator[Chunk]:
        # The document's own numbering ("Table 3") is in the title; table_index stays in metadata.
        label = "[Table" + (f", page {table.location.page}]" if table.location.page else "]")
        heading = f"{label} {table.title}".strip()
        header = _markdown_row(table.header) + "\n" + _markdown_row(("---",) * len(table.header))
        row_lines = [_markdown_row(_pad(row, len(table.header))) for row in table.rows]

        groups = _row_groups(row_lines, budget=self._chunk_size - len(heading) - len(header) - 2)
        for part, (start, end) in enumerate(groups, 1):
            body = "\n".join(row_lines[start:end])
            text = f"{heading}\n{header}" + (f"\n{body}" if body else "")
            yield self._chunk(
                text, "table", table, part, len(groups),
                table_index=table.table_index, table_title=table.title or None,
                columns=" | ".join(table.header), row_count=len(table.rows),
                column_count=len(table.header),
                row_start=start + 1 if body else None, row_end=end if body else None,
            )

    # ---- images ----

    def _image_chunks(self, image: ImageElement) -> Iterator[Chunk]:
        description = self._vision.descriptions.get(image.image_hash)
        if description is None or description.is_decorative:
            return
        label = "[Figure" + (f", page {image.location.page}]" if image.location.page else "]")
        text = f"{label} {image.caption}".strip() + f"\n{description.text}"
        chunk = self._chunk(
            text, "image", image, 1, 1,
            image_index=image.image_index, image_source=image.source,
            content_type=description.content_type, caption=image.caption or None,
            figure_label=figure_label(image.caption), width=image.width, height=image.height,
            image_hash=image.image_hash, vision_model=self._vision_model,
            prompt_version=PROMPT_VERSION, embedding_input=self._embedding_input,
        )
        yield Chunk(document=chunk.document, image=image)

    # ---- shared ----

    def _chunk(self, text: str, chunk_type: str, element: Element, part: int, parts: int, **extra) -> Chunk:
        location = element.location
        metadata = ChunkMetadata(
            file_name=self._file.file_name,
            file_type=self._file.file_type,
            title=self._file.file_name,
            file_hash=self._file.file_hash,
            chunk_id=make_chunk_id(self._file.file_hash, location.order, part),
            chunk_type=chunk_type,
            chunk_index=self._chunk_index,
            page=location.page,
            total_pages=self._file.total_pages,
            section=location.section or None,
            section_path=location.section_path or None,
            # A text chunk joins several blocks, so a single block's box would mislead.
            bbox=list(location.bbox) if location.bbox and chunk_type != "text" else None,
            part=part,
            parts=parts,
            extractor=self._file.extractor,
            ingested_at=self._file.ingested_at,
            **extra,
        )
        self._chunk_index += 1
        return Chunk(document=Document(page_content=text, metadata=metadata.to_metadata()))


def _markdown_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def _pad(row: tuple[str, ...], width: int) -> tuple[str, ...]:
    return (row + ("",) * width)[:width]


def _row_groups(row_lines: list[str], budget: int) -> list[tuple[int, int]]:
    """Greedy split of rows into [start, end) ranges whose text fits ``budget`` (at least one row each)."""
    if not row_lines:
        return [(0, 0)]
    groups, start, size = [], 0, 0
    for index, line in enumerate(row_lines):
        if index > start and size + len(line) + 1 > budget:
            groups.append((start, index))
            start, size = index, 0
        size += len(line) + 1
    groups.append((start, len(row_lines)))
    return groups
