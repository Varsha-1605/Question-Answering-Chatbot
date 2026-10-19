import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from app.injestion.models import FILE_TYPE, IngestRequestDto, RequestStatusEnum
from app.injestion.utils import load_file_and_push_to_database
from app.settings import config_settings
from app.vector_db.utils import close_vector_db


async def reindex(folder: Path, index_name: str, namespace: str, concurrency: int) -> int:
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lstrip(".").lower() in FILE_TYPE)
    if not files:
        logger.warning(f"No supported files ({', '.join(FILE_TYPE)}) in {folder}")
        return 0

    slots = asyncio.Semaphore(concurrency)

    async def ingest(path: Path) -> bool:
        async with slots:
            result = await load_file_and_push_to_database(IngestRequestDto(
                pre_signed_url=str(path), file_name=path.name, original_file_name=path.name,
                file_type=path.suffix.lstrip(".").lower(), index_name=index_name,
                namespace=namespace, delete_status=True,
            ))
        ok = result.status == RequestStatusEnum.COMPLETED
        log = logger.info if ok and not result.error_detail else logger.warning
        log(f"{path.name}: {result.status.value}, {result.total_pages} chunks"
            + (f" ({result.error_detail})" if result.error_detail else ""))
        return ok

    try:
        results = await asyncio.gather(*(ingest(path) for path in files))
    finally:
        close_vector_db()
    failed = results.count(False)
    logger.info(f"Re-indexed {len(files) - failed}/{len(files)} files into {index_name}/{namespace}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ingest a folder of documents into an index.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--index", default=config_settings.INDEX_NAME, help="default: INDEX_NAME")
    parser.add_argument("--namespace", default=config_settings.DEFAULT_NAMESPACE, help="default: DEFAULT_NAMESPACE")
    parser.add_argument("--concurrency", type=int, default=2, help="files processed at the same time")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")
    failed = asyncio.run(reindex(args.folder, args.index, args.namespace, max(1, args.concurrency)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
