from dataclasses import dataclass, field
from itertools import count
from typing import Literal, Optional

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.injestion.multimodal.elements import (
    ExtractedDocument,
    ExtractionLimits,
    ImageElement,
    Location,
    TableElement,
    TextElement,
)
from app.injestion.multimodal.images import (
    CAPTION_PREFIXES,
    ImageBudget,
    PreparedImage,
    image_hash,
    nearby_text,
    prepare_image,
)

TITLE_STYLE = "Title"
HEADING_STYLE_PREFIX = "Heading"
CAPTION_STYLE = "Caption"
TABLE_TITLE_PREFIX = "table"
# Image references: modern DrawingML (a:blip r:embed) and legacy VML (v:imagedata r:id).
BLIP_TAG = qn("a:blip")
BLIP_REL = qn("r:embed")
VML_IMAGE_TAG = "{urn:schemas-microsoft-com:vml}imagedata"  # "v" is not in python-docx's nsmap
VML_IMAGE_REL = qn("r:id")


@dataclass(slots=True)
class _Item:
    kind: Literal["text", "table", "image"]
    section: str
    section_path: str
    heading_level: Optional[int]
    text: str = ""
    is_caption: bool = False
    rows: list[tuple[str, ...]] = field(default_factory=list)
    title: str = ""
    image_rel_id: str = ""


def extract_docx(path: str, limits: ExtractionLimits) -> ExtractedDocument:
    document = docx.Document(path)
    items = _walk_body(document)
    return _to_elements(document, items, limits)


def _walk_body(document: DocxDocument) -> list[_Item]:
    """First pass: body order, section tracking, images linked to their paragraph."""
    items: list[_Item] = []
    title = ""
    headings: list[str] = []  # headings[i] is the current level-(i+1) heading

    for child in document.element.body.iterchildren():
        section = headings[-1] if headings else title
        section_path = " > ".join(part for part in (title, *headings) if part)
        level = len(headings) or None

        if child.tag == qn("w:tbl"):
            rows = _table_rows(Table(child, document))
            if not rows:
                continue
            table_title = ""
            if items and items[-1].kind == "text" and items[-1].text.lower().startswith(TABLE_TITLE_PREFIX):
                table_title = items.pop().text  # stored with the table, not twice
            elif section.lower().startswith(TABLE_TITLE_PREFIX):
                table_title = section  # a heading such as "Table 6. Payroll Register" names the table
            items.append(_Item("table", section, section_path, level, rows=rows, title=table_title))
            continue

        if child.tag != qn("w:p"):
            continue
        paragraph = Paragraph(child, document)
        style = paragraph.style.name if paragraph.style is not None else ""
        text = " ".join(paragraph.text.split())

        heading_level = _heading_level(style)
        if style == TITLE_STYLE and text:
            title, headings = text, []
            continue
        if heading_level and text:
            headings = headings[: heading_level - 1] + [text]
            continue

        for rel_id in _image_rel_ids(child):
            items.append(_Item("image", section, section_path, level, image_rel_id=rel_id))
        if text:
            is_caption = style == CAPTION_STYLE or text.lower().startswith(CAPTION_PREFIXES)
            items.append(_Item("text", section, section_path, level, text=text, is_caption=is_caption))
    return items


def _to_elements(document: DocxDocument, items: list[_Item], limits: ExtractionLimits) -> ExtractedDocument:
    """Second pass: decode images, attach captions and nearby text, emit elements in order."""
    result = ExtractedDocument()
    budget = ImageBudget(limits.max_images)
    order, table_index, image_index = count(), count(1), count(1)
    before, after = nearby_text([item.text if item.kind == "text" else "" for item in items])
    # Matched up front so a caption placed before its image is not also emitted as body text.
    captions = {
        position: caption
        for position, item in enumerate(items)
        if item.kind == "image" and (caption := _caption_position(items, position)) is not None
    }
    caption_positions = set(captions.values())

    for position, item in enumerate(items):
        if position in caption_positions:
            continue
        location = Location(order=next(order), section=item.section, section_path=item.section_path)

        if item.kind == "text":
            result.elements.append(TextElement(text=item.text, location=location, heading_level=item.heading_level))
        elif item.kind == "table":
            header, *body = item.rows
            result.elements.append(TableElement(
                header=header, rows=tuple(body), location=location,
                title=item.title, table_index=next(table_index),
            ))
        elif decoded := _decode_image(document, item.image_rel_id, budget, limits):
            prepared, digest = decoded
            caption_position = captions.get(position)
            result.elements.append(ImageElement(
                image_bytes=prepared.data, mime_type=prepared.mime_type,
                width=prepared.width, height=prepared.height,
                image_hash=digest, source="docx_inline", location=location,
                caption=items[caption_position].text if caption_position is not None else "",
                context=" ".join(part for part in (before[position], after[position]) if part),
                image_index=next(image_index),
            ))

    result.images_skipped = budget.skipped
    return result


def _decode_image(
    document: DocxDocument, rel_id: str, budget: ImageBudget, limits: ExtractionLimits
) -> Optional[tuple[PreparedImage, str]]:
    """Decode one inline image; None (counted as skipped) if missing, repeated, tiny or unreadable."""
    relationship = document.part.rels.get(rel_id)
    if relationship is None or relationship.is_external:
        budget.reject()
        return None
    blob = relationship.target_part.blob
    digest = image_hash(blob)
    if budget.should_skip(digest):
        return None
    extension = str(relationship.target_part.partname).rsplit(".", 1)[-1]
    prepared = prepare_image(blob, extension, limits)
    if prepared is None:
        budget.reject(digest)
        return None
    budget.add(digest)
    return prepared, digest


def _caption_position(items: list[_Item], position: int) -> Optional[int]:
    """The caption paragraph right after the image, else right before it."""
    for neighbour in (position + 1, position - 1):
        if 0 <= neighbour < len(items) and items[neighbour].kind == "text" and items[neighbour].is_caption:
            return neighbour
    return None


def _heading_level(style: str) -> Optional[int]:
    if not style.startswith(HEADING_STYLE_PREFIX):
        return None
    suffix = style[len(HEADING_STYLE_PREFIX):].strip()
    return int(suffix) if suffix.isdigit() else None


def _image_rel_ids(paragraph_element) -> list[str]:
    rel_ids = [blip.get(BLIP_REL) for blip in paragraph_element.iter(BLIP_TAG)]
    rel_ids += [image.get(VML_IMAGE_REL) for image in paragraph_element.iter(VML_IMAGE_TAG)]
    return [rel_id for rel_id in rel_ids if rel_id]


def _table_rows(table: Table) -> list[tuple[str, ...]]:
    rows = [tuple(" ".join(cell.text.split()) for cell in row.cells) for row in table.rows]
    return [row for row in rows if any(row)]
