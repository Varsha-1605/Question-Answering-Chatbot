import ipaddress
import re
import socket
import requests

from loguru import logger
from os import path, remove
from urllib.parse import urlparse
from os.path import expanduser, isfile
from tempfile import NamedTemporaryFile
from typing import Any, IO, Dict, List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader
)

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from ai_workflow.settings import config_settings
from ai_workflow.injestion.models import FILE_TYPE


def split_text(
        text: list[Document],
        chunk_size: int,
        chunk_overlap: int
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(text)


def assert_public_http_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to non-public addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Download URL must be an absolute http(s) URL")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve download host: {parsed.hostname}") from e
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Download URL resolves to a non-public address")


def download_file_from_url(url: str, dest_path: str, timeout: int = 60) -> bool:
    try:
        assert_public_http_url(url)
        logger.info(f"Downloading file from URL: {url[:100]}...")
        # No redirects: they could point to an internal address.
        response = requests.get(url, timeout=timeout, stream=True, allow_redirects=False)
        if response.is_redirect:
            raise ValueError("Download URL redirected; redirects are not followed")
        response.raise_for_status()

        max_bytes = config_settings.MAX_FILE_SIZE_MB * 1024 * 1024
        declared_size = int(response.headers.get("Content-Length") or 0)
        if declared_size > max_bytes:
            raise ValueError(f"File is larger than the {config_settings.MAX_FILE_SIZE_MB} MB limit")

        # The header may be missing or wrong, so also check while streaming.
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(
                            f"File is larger than the {config_settings.MAX_FILE_SIZE_MB} MB limit"
                        )
                    f.write(chunk)

        logger.info(f"Successfully downloaded file to: {dest_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download file from URL: {str(e)}")
        raise ValueError(f"Failed to download file from URL: {str(e)}")

class URLDownloaderMixin:
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        except Exception:
            return False

    def __init__(self, file_path: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        self._temp_file: Optional[IO[bytes]] = None
        self._original_url: Optional[str] = None
        self.file_path = str(file_path) if file_path else ""

        if "~" in self.file_path:
            self.file_path = expanduser(self.file_path)

        if isfile(self.file_path):
            return

        if self._is_valid_url(self.file_path):
            self._original_url = self.file_path
            self._temp_file = NamedTemporaryFile(delete=False, suffix=self._get_suffix_from_url())
            self._temp_file.close()
            download_file_from_url(self._original_url, self._temp_file.name)
            self.file_path = self._temp_file.name
            return

        raise ValueError(f"Invalid file path or URL: {self.file_path}")

    def _get_suffix_from_url(self) -> str:
        if self._original_url:
            parsed = urlparse(self._original_url)
            path_part = parsed.path
            if '.' in path_part:
                ext = path_part.rsplit('.', 1)[-1].lower()
                if ext in FILE_TYPE:
                    return f".{ext}"
        return ""

    def __del__(self) -> None:
        if hasattr(self, '_temp_file') and self._temp_file:
            try:
                if hasattr(self._temp_file, 'name') and path.exists(self._temp_file.name):
                    remove(self._temp_file.name)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")


class FileLoader(URLDownloaderMixin, BaseLoader):
    LOADER_MAP = {
        "txt": TextLoader,
        "pdf": PyMuPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
        "xlsx": UnstructuredExcelLoader,
        "csv": UnstructuredCSVLoader
    }

    def __init__(self, file_path: str, process_type: str = "txt") -> None:
        self.process_type = process_type
        self._validate_process_type()
        URLDownloaderMixin.__init__(self, file_path=file_path)

    def _validate_process_type(self) -> None:
        if self.process_type not in FILE_TYPE:
            raise ValueError(f"Unsupported process type: {self.process_type}. Allowed types: {', '.join(FILE_TYPE)}")

    def _validate_file_path(self) -> None:
        if not path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self) -> List[Document]:
        try:
            self._validate_file_path()
            loader_cls = self.LOADER_MAP.get(self.process_type)
            if not loader_cls:
                raise ValueError(f"Unsupported process type: {self.process_type}")

            loader = loader_cls(file_path=self.file_path)
            file_contents = loader.load()

            logger.info(
                f"Successfully loaded {self.process_type} file: {self.file_path} ({len(file_contents)} documents)")
            return file_contents

        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {str(e)}")
            raise


def clean_transcript(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\<\>\~\`]', '', text)
    return text.strip()


@traceable(
    name="load_file",
    run_type="tool",
    # Log only the path; signed URLs contain credentials.
    process_inputs=lambda inputs: {
        "file": urlparse(inputs["file_path"]).path or inputs["file_path"],
        "file_type": inputs["file_type"],
    },
    process_outputs=lambda documents: {
        "documents": len(documents),
        "characters": sum(len(document.page_content) for document in documents),
    },
)
def load_documents(file_path: str, file_type: str) -> List[Document]:
    return FileLoader(file_path, process_type=file_type).load()


@traceable(
    name="clean_and_chunk",
    run_type="tool",
    process_inputs=lambda inputs: {
        "documents": len(inputs["documents"]),
        "chunk_size": inputs["chunk_size"],
        "chunk_overlap": inputs["chunk_overlap"],
    },
    process_outputs=lambda chunks: {
        "chunks": len(chunks),
        "average_characters": round(sum(len(chunk.page_content) for chunk in chunks) / len(chunks))
        if chunks
        else 0,
    },
)
def clean_and_chunk(
        documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    cleaned_documents = [
        Document(page_content=clean_transcript(doc.page_content), metadata=doc.metadata)
        for doc in documents
    ]
    return split_text(
        text=cleaned_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def file_loader(
        pre_signed_url: str,
        file_name: str,
        file_type: str,
        metadata: Optional[List[Dict[str, str]]] = None
) -> List[Document]:
    try:
        if file_type not in FILE_TYPE:
            raise ValueError(f"Unsupported file type: {file_type}")

        loaded_documents = load_documents(pre_signed_url, file_type)

        logger.debug(f"Loaded {len(loaded_documents)} documents before chunking")

        chunks_documents = clean_and_chunk(
            loaded_documents,
            chunk_size=config_settings.CHUNK_SIZE,
            chunk_overlap=config_settings.CHUNK_OVERLAP,
        )

        logger.info(f"Document processing complete: {len(chunks_documents)} chunks created")

        base_metadata = {
            "file_name": file_name,
            "file_type": file_type,
        }

        if metadata:
            for meta in metadata:
                base_metadata.update(meta)

        for document in chunks_documents:
            document.metadata.update(base_metadata)
            document.metadata["title"] = document.metadata.get("title") or file_name

        return chunks_documents

    except Exception as e:
        logger.error(f"Error in file_loader: {str(e)}")
        raise