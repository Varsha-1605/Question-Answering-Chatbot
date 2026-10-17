import socket

import pytest
from langchain_core.documents import Document

from ai_workflow.injestion import doc_loader
from ai_workflow.injestion.doc_loader import (
    assert_public_http_url,
    clean_and_chunk,
    clean_transcript,
    file_loader,
)


def test_clean_transcript_collapses_whitespace():
    assert clean_transcript("a \n\t b   c ") == "a b c"


def test_clean_transcript_handles_empty_input():
    assert clean_transcript("") == ""


def test_chunks_respect_size_and_keep_metadata():
    document = Document(page_content="word " * 200, metadata={"page": 3})
    chunks = clean_and_chunk([document], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)


def test_file_loader_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        file_loader("http://example.com/x.exe", "x.exe", "exe")


def test_file_loader_adds_file_metadata_and_title(monkeypatch):
    monkeypatch.setattr(
        doc_loader, "load_documents", lambda path, file_type: [Document(page_content="hello world")]
    )
    chunks = file_loader("http://example.com/a.txt", "a.txt", "txt", metadata=[{"team": "x"}])
    assert chunks[0].metadata["file_name"] == "a.txt"
    assert chunks[0].metadata["title"] == "a.txt"
    assert chunks[0].metadata["team"] == "x"


@pytest.mark.parametrize("url", ["ftp://example.com/a", "file:///etc/passwd", "/local/path"])
def test_ssrf_guard_rejects_non_http_urls(url):
    with pytest.raises(ValueError):
        assert_public_http_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "169.254.169.254"])
def test_ssrf_guard_rejects_hosts_resolving_to_private_addresses(monkeypatch, address):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", (address, 443))]
    )
    with pytest.raises(ValueError, match="non-public"):
        assert_public_http_url("https://internal.example.com/file.pdf")


def test_ssrf_guard_accepts_public_address(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", ("8.8.8.8", 443))]
    )
    assert_public_http_url("https://example.com/file.pdf")
