from dataclasses import dataclass
from itertools import count
from typing import Optional

import pymupdf
from loguru import logger

from app.injestion.multimodal.elements import (
    ExtractedDocument,
    ExtractionLimits,
    ImageElement,
    ImageSource,
    Location,
    TableElement,
    TextElement,
)
from app.injestion.multimodal.images import (
    ImageBudget,
    PreparedImage,
    CAPTION_PREFIXES,
    image_hash,
    nearby_text,
    prepare_image,
    render_region,
)

TABLE_TITLE_PREFIX = "table"
# A page with less text than this and at least one image is treated as scanned.
SCANNED_PAGE_MAX_TEXT_CHARS = 20
# Vector clusters smaller than this (points) are rules, underlines or bullets.
MIN_FIGURE_SIDE_PT = 40
# How far (points) a caption or table title may sit from its figure or table.
CAPTION_MAX_GAP_PT = 60
# Grid-like drawings (heatmaps, attention maps) fool find_tables. A real table
# has a reasonable number of columns and most of its cells filled.
MAX_TABLE_COLUMNS = 20
MIN_TABLE_FILL_RATIO = 0.5


@dataclass(slots=True)
class _Block:
    rect: pymupdf.Rect
    text: str


@dataclass(slots=True)
class _Table:
    rect: pymupdf.Rect
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(slots=True)
class _Figure:
    rect: pymupdf.Rect
    image: PreparedImage
    digest: str
    source: ImageSource


def extract_pdf(path: str, limits: ExtractionLimits) -> ExtractedDocument:
    return _PdfExtraction(limits).run(path)


class _PdfExtraction:
    def __init__(self, limits: ExtractionLimits) -> None:
        self._limits = limits
        self._budget = ImageBudget(limits.max_images)
        self._order = count()
        self._table_index = count(1)
        self._image_index = count(1)
        self._result = ExtractedDocument()

    def run(self, path: str) -> ExtractedDocument:
        with pymupdf.open(path) as pdf:
            self._result.total_pages = len(pdf)
            for page in pdf:
                self._extract_page(pdf, page)
        self._result.images_skipped = self._budget.skipped
        return self._result

    # ---- per page ----

    def _extract_page(self, pdf: pymupdf.Document, page: pymupdf.Page) -> None:
        blocks = _text_blocks(page)
        tables = _find_tables(page)
        text_chars = sum(len(block.text) for block in blocks)

        if text_chars < SCANNED_PAGE_MAX_TEXT_CHARS and not tables:
            figures = self._scanned_page(page)
        else:
            figures = self._raster_figures(pdf, page)
            figures += self._vector_figures(page, occupied=[t.rect for t in tables] + [f.rect for f in figures])

        captions = _match_captions([f.rect for f in figures], blocks, CAPTION_PREFIXES)
        titles = _match_captions([t.rect for t in tables], blocks, (TABLE_TITLE_PREFIX,), above_only=True)
        used = {id(block) for block in (*captions, *titles) if block is not None}
        # Text inside tables and raster images is dropped (stored by those elements).
        # Text inside vector figures is kept: a borderless table that find_tables
        # missed can look like a vector figure, and its text must not be lost.
        covered = [t.rect for t in tables] + [f.rect for f in figures if f.source != "vector_figure"]
        body = [b for b in blocks if id(b) not in used and not _inside_any(b.rect, covered)]

        self._emit_in_reading_order(page.number + 1, body, tables, titles, figures, captions)

    def _emit_in_reading_order(self, page_no, body, tables, titles, figures, captions) -> None:
        """Emit elements top-to-bottom; each figure gets the text just before and after it as context."""
        items = (
            [(b.rect, "text", b, None) for b in body]
            + [(t.rect, "table", t, title) for t, title in zip(tables, titles, strict=True)]
            + [(f.rect, "figure", f, caption) for f, caption in zip(figures, captions, strict=True)]
        )
        items.sort(key=lambda item: (round(item[0].y0), item[0].x0))

        before, after = nearby_text([item[2].text if item[1] == "text" else "" for item in items])

        for position, (rect, kind, payload, label) in enumerate(items):
            location = Location(order=next(self._order), page=page_no, bbox=tuple(rect))
            if kind == "text":
                self._result.elements.append(TextElement(text=payload.text, location=location))
            elif kind == "table":
                self._result.elements.append(TableElement(
                    header=payload.header, rows=payload.rows, location=location,
                    title=label.text if label else "", table_index=next(self._table_index),
                ))
            else:
                self._result.elements.append(ImageElement(
                    image_bytes=payload.image.data, mime_type=payload.image.mime_type,
                    width=payload.image.width, height=payload.image.height,
                    image_hash=payload.digest, source=payload.source, location=location,
                    caption=label.text if label else "",
                    context=" ".join(part for part in (before[position], after[position]) if part),
                    image_index=next(self._image_index),
                ))

    # ---- figures ----

    def _scanned_page(self, page: pymupdf.Page) -> list[_Figure]:
        if not page.get_image_info():
            return []  # blank page
        figure = self._render(page, page.rect, "scanned_page")
        return [figure] if figure else []

    def _raster_figures(self, pdf: pymupdf.Document, page: pymupdf.Page) -> list[_Figure]:
        figures: list[_Figure] = []
        for info in page.get_image_info(xrefs=True):
            rect = pymupdf.Rect(info["bbox"])
            if not info["xref"]:  # inline image without an object: render its area instead
                figure = self._render(page, rect, "embedded")
            else:
                figure = self._embedded(pdf, info["xref"], rect)
            if figure:
                figures.append(figure)
        return figures

    def _embedded(self, pdf: pymupdf.Document, xref: int, rect: pymupdf.Rect) -> Optional[_Figure]:
        extracted = pdf.extract_image(xref)
        if not extracted:
            self._budget.reject()
            return None
        digest = image_hash(extracted["image"])
        if self._budget.should_skip(digest):
            return None
        prepared = prepare_image(extracted["image"], extracted.get("ext", "png"), self._limits)
        if prepared is None:
            self._budget.reject(digest)
            return None
        self._budget.add(digest)
        return _Figure(rect, prepared, digest, "embedded")

    def _vector_figures(self, page: pymupdf.Page, occupied: list[pymupdf.Rect]) -> list[_Figure]:
        if self._budget.is_full():
            return []
        figures: list[_Figure] = []
        for rect in page.cluster_drawings():
            rect = pymupdf.Rect(rect)
            # Small clusters are rules/underlines; overlapping ones are table borders or image frames.
            if min(rect.width, rect.height) < MIN_FIGURE_SIDE_PT or any(rect.intersects(o) for o in occupied):
                continue
            figure = self._render(page, rect, "vector_figure")
            if figure:
                figures.append(figure)
        return figures

    def _render(self, page: pymupdf.Page, rect: pymupdf.Rect, source: ImageSource) -> Optional[_Figure]:
        if self._budget.is_full():
            self._budget.reject()
            return None
        prepared = render_region(page, rect, self._limits)
        if prepared is None:
            self._budget.reject()
            return None
        digest = image_hash(prepared.data)
        if self._budget.should_skip(digest):
            return None
        self._budget.add(digest)
        return _Figure(rect, prepared, digest, source)


# ---- page helpers (pure functions) ----

def _text_blocks(page: pymupdf.Page) -> list[_Block]:
    blocks = []
    for x0, y0, x1, y1, text, _number, block_type in page.get_text("blocks", sort=True):
        text = " ".join(text.split())  # PDF line breaks inside a block are layout, not meaning
        if block_type == 0 and text:
            blocks.append(_Block(pymupdf.Rect(x0, y0, x1, y1), text))
    return blocks


def _find_tables(page: pymupdf.Page) -> list[_Table]:
    try:
        found = page.find_tables().tables
    except Exception as error:  # noqa: BLE001 - layout heuristics can fail on odd pages; keep the text
        logger.warning(f"Table detection failed on page {page.number + 1}: {error}")
        return []

    tables = []
    for table in found:
        rows = [tuple(_clean_cell(cell) for cell in row) for row in table.extract()]
        rows = [row for row in rows if any(row)]
        if not _looks_like_table(rows):
            continue  # left to the figure detector instead
        if table.header.external:  # header drawn above the table body
            header, body = tuple(_clean_cell(name) for name in table.header.names), rows
        else:
            header, body = rows[0], rows[1:]
        tables.append(_Table(pymupdf.Rect(table.bbox), header, tuple(body)))
    return tables


def _looks_like_table(rows: list[tuple[str, ...]]) -> bool:
    if not rows or len(rows[0]) > MAX_TABLE_COLUMNS:
        return False
    cells = [cell for row in rows for cell in row]
    return sum(1 for cell in cells if cell) / len(cells) >= MIN_TABLE_FILL_RATIO


def _clean_cell(value: Optional[str]) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _inside_any(rect: pymupdf.Rect, regions: list[pymupdf.Rect]) -> bool:
    center = pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    return any(center in region for region in regions)


def _match_captions(
    targets: list[pymupdf.Rect], blocks: list[_Block], prefixes: tuple[str, ...], above_only: bool = False
) -> list[Optional[_Block]]:
    """For each target, the closest block starting with one of ``prefixes`` (below or above it).

    A caption may serve several targets: multi-panel figures share one caption.
    """
    candidates = [b for b in blocks if b.text.lower().startswith(prefixes)]
    matches: list[Optional[_Block]] = []
    for target in targets:
        best, best_gap = None, CAPTION_MAX_GAP_PT
        for block in candidates:
            gap_above = target.y0 - block.rect.y1
            if above_only and gap_above < 0:
                continue  # titles must sit above the table
            gap = gap_above if above_only else min(abs(block.rect.y0 - target.y1), abs(gap_above))
            if gap <= best_gap:
                best, best_gap = block, gap
        matches.append(best)
    return matches
