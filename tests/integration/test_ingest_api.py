"""Integration tests for the POST /api/v1/ingest endpoint."""
from __future__ import annotations

import io
import pytest
from unittest.mock import MagicMock, patch


def _make_upload(content: bytes, filename: str, content_type: str = "text/plain"):
    return ("file", (filename, io.BytesIO(content), content_type))


@pytest.fixture
def patched_pipeline():
    """Patch all external calls for ingest: parse, chunk, embed, Qdrant."""
    with (
        patch("app.api.routes.ingest.parse_document", return_value="# Doc\n\nContent here.") as mock_parse,
        patch("app.api.routes.ingest.semantic_chunk", return_value=["Chunk one.", "Chunk two."]) as mock_chunk,
        patch("app.api.routes.ingest.embed_texts", return_value=[[0.1]*768, [0.2]*768]) as mock_embed,
        patch("app.api.routes.ingest.ensure_collection", return_value="rag_knowledge_base") as mock_col,
        patch("app.api.routes.ingest.upsert_chunks", return_value=2) as mock_upsert,
    ):
        yield {
            "parse": mock_parse,
            "chunk": mock_chunk,
            "embed": mock_embed,
            "col": mock_col,
            "upsert": mock_upsert,
        }


class TestIngestEndpoint:
    def test_ingest_txt_returns_201(self, test_client, patched_pipeline):
        response = test_client.post(
            "/api/v1/ingest",
            files=[_make_upload(b"Sample text content.", "sample.txt")],
        )
        assert response.status_code == 201
        body = response.json()
        assert body["filename"] == "sample.txt"
        assert body["total_chunks"] == 2
        assert body["file_type"] == "txt"
        assert "document_id" in body

    def test_ingest_md_returns_201(self, test_client, patched_pipeline):
        response = test_client.post(
            "/api/v1/ingest",
            files=[_make_upload(b"# Header\nContent.", "notes.md", "text/markdown")],
        )
        assert response.status_code == 201
        assert response.json()["file_type"] == "md"

    def test_unsupported_extension_returns_415(self, test_client):
        response = test_client.post(
            "/api/v1/ingest",
            files=[_make_upload(b"<html></html>", "page.html", "text/html")],
        )
        assert response.status_code == 415

    def test_empty_file_returns_422(self, test_client, patched_pipeline):
        patched_pipeline["chunk"].return_value = []
        response = test_client.post(
            "/api/v1/ingest",
            files=[_make_upload(b"   ", "empty.txt")],
        )
        assert response.status_code == 422

    def test_missing_file_returns_422(self, test_client):
        response = test_client.post("/api/v1/ingest")
        assert response.status_code == 422

    def test_response_contains_collection_name(self, test_client, patched_pipeline):
        response = test_client.post(
            "/api/v1/ingest",
            files=[_make_upload(b"Content.", "doc.txt")],
        )
        assert response.status_code == 201
        assert response.json()["collection"] == "rag_knowledge_base"
