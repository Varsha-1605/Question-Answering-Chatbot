import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from loguru import logger
from weaviate.classes.query import Filter

from app.settings import config_settings
from app.vector_db.sanitization import sanitize_index_name, sanitize_namespace
from app.vector_db.utils import close_vector_db
from app.vector_db.weaviate_utils.weaviate_client import FILE_NAME_PROPERTY, db_connection

DEFAULT_OUT = Path("evals/chunks_dump.json")
LEGACY_CHUNK_TYPE = "text (legacy)"  # chunks from the text-only loader have no chunk_type


def fetch_chunks(index_name: str, namespace: str, file_name: Optional[str]) -> list[dict]:
    """One record per stored chunk: id, chunk text and metadata."""
    collection = db_connection.get_client().collections.get(sanitize_index_name(index_name))
    tenant = collection.with_tenant(sanitize_namespace(namespace))

    if file_name:
        objects = tenant.query.fetch_objects(
            filters=Filter.by_property(FILE_NAME_PROPERTY).equal(file_name),
            limit=10_000,
        ).objects
    else:
        objects = tenant.iterator(include_vector=False)

    chunks = []
    for obj in objects:
        # drop empty fields: each chunk type fills only its own metadata group
        meta = {key: value for key, value in obj.properties.items() if value is not None}
        text = meta.pop(config_settings.WEAVIATE_TEXT_KEY, "") or ""
        chunks.append({"id": str(obj.uuid), "text": text, "meta": meta})
    return chunks


def reading_order(chunk: dict) -> tuple:
    """Group by file, then page and chunk order, so related chunks sit together."""
    meta = chunk["meta"]
    return (str(meta.get(FILE_NAME_PROPERTY, "")), meta.get("page") or 0,
            meta.get("chunk_index") or 0, meta.get("part") or 1)


def dump(index_name: str, namespace: str, file_name: Optional[str], out: Path) -> int:
    if not db_connection.tenant_exists(index_name, namespace):
        logger.error(f"No namespace {namespace!r} in index {index_name!r}; ingest documents first")
        return 0

    chunks = sorted(fetch_chunks(index_name, namespace, file_name), key=reading_order)
    if not chunks:
        logger.warning(f"No chunks found in {index_name}/{namespace}"
                       + (f" for {file_name}" if file_name else ""))
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"Dumped {len(chunks)} chunks from {index_name}/{namespace} to {out}")

    counts = Counter((str(c["meta"].get(FILE_NAME_PROPERTY, "?")),
                      str(c["meta"].get("chunk_type", LEGACY_CHUNK_TYPE))) for c in chunks)
    for (name, chunk_type), count in sorted(counts.items()):
        logger.info(f"  {name:<45} {chunk_type:<14} {count}")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump the stored chunks of an index/namespace to JSON.")
    parser.add_argument("--index", default=config_settings.INDEX_NAME, help="default: INDEX_NAME")
    parser.add_argument("--namespace", default=config_settings.DEFAULT_NAMESPACE, help="default: DEFAULT_NAMESPACE")
    parser.add_argument("--file", dest="file_name", help="only this file (exact stored file_name)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    args = parser.parse_args()

    try:
        count = dump(args.index, args.namespace, args.file_name, args.out)
    finally:
        close_vector_db()
    sys.exit(0 if count else 1)


if __name__ == "__main__":
    main()
