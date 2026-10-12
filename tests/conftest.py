"""
pytest configuration and shared fixtures for the RAG Knowledge Assistant test suite.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session")
def test_client():
    """
    Return a synchronous TestClient for the FastAPI app.
    Patches Qdrant and LLM providers so no external services are needed.
    """
    with (
        patch("app.vectorstore.client.get_qdrant_client") as mock_qdrant,
        patch("app.embeddings.mpnet.get_embedding_model") as mock_model,
    ):
        # Mock Qdrant client
        mock_qdrant.return_value = MagicMock()
        # Mock embedding model
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [[0.1] * 768]
        mock_model.return_value = mock_instance

        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return b"This is a sample document about Python programming. Python is a versatile language."


@pytest.fixture
def sample_md_bytes() -> bytes:
    return b"# Machine Learning\n\nMachine learning is a subset of artificial intelligence."


@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    return [[0.1] * 768, [0.2] * 768, [0.3] * 768]


@pytest.fixture
def mock_search_results() -> list[dict]:
    return [
        {
            "document_id": "doc-001",
            "filename": "sample.txt",
            "chunk_index": 0,
            "total_chunks": 3,
            "text": "Python is a versatile programming language.",
            "file_type": "txt",
            "ingested_at": "2024-01-01T00:00:00Z",
            "_score": 0.92,
            "page_number": None,
        }
    ]
