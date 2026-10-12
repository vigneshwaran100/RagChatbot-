"""Integration tests for the POST /api/v1/query endpoint."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


VALID_REFORMULATION = json.dumps({
    "refined_query": "What is Python programming language?",
    "search_terms": ["Python", "programming"],
    "intent": "Learn about Python.",
    "requires_multiple_chunks": False,
})

MOCK_SEARCH_RESULT = {
    "_score": 0.92,
    "document_id": "doc-001",
    "filename": "python_guide.txt",
    "chunk_index": 0,
    "total_chunks": 5,
    "text": "Python is a high-level, interpreted programming language.",
    "file_type": "txt",
    "ingested_at": "2024-01-01T00:00:00Z",
    "page_number": None,
}


@pytest.fixture
def patched_query_pipeline(monkeypatch):
    with (
        patch("app.api.routes.query.reformulate_query") as mock_reform,
        patch("app.api.routes.query.ensure_collection", return_value="rag_knowledge_base"),
        patch("app.api.routes.query.search_vectors", return_value=[MOCK_SEARCH_RESULT]) as mock_search,
        patch("app.api.routes.query.call_llm", return_value=("Python is a high-level language.", "groq")) as mock_llm,
    ):
        from app.schemas.reformulation import ReformulatedQuery
        mock_reform.return_value = ReformulatedQuery(
            refined_query="What is Python programming language?",
            search_terms=["Python"],
            intent="Learn about Python.",
            requires_multiple_chunks=False,
        )
        yield {
            "reform": mock_reform,
            "search": mock_search,
            "llm": mock_llm,
        }


@pytest.fixture
def patched_unanswerable_pipeline():
    with (
        patch("app.api.routes.query.reformulate_query") as mock_reform,
        patch("app.api.routes.query.ensure_collection", return_value="rag_knowledge_base"),
        patch("app.api.routes.query.search_vectors", return_value=[]) as mock_search,
    ):
        from app.schemas.reformulation import ReformulatedQuery
        mock_reform.return_value = ReformulatedQuery(
            refined_query="What is the secret of life?",
            search_terms=["secret", "life"],
            intent="Find the meaning of life.",
            requires_multiple_chunks=False,
        )
        yield


class TestQueryEndpoint:
    def test_answerable_question_returns_200(self, test_client, patched_query_pipeline):
        response = test_client.post(
            "/api/v1/query",
            json={"question": "What is Python?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["is_answerable"] is True
        assert len(body["answer"]) > 0
        assert len(body["evidence"]) >= 1
        assert body["evidence"][0]["filename"] == "python_guide.txt"
        assert body["provider_used"] == "groq"

    def test_unanswerable_question_returns_200_with_flag(
        self, test_client, patched_unanswerable_pipeline
    ):
        response = test_client.post(
            "/api/v1/query",
            json={"question": "What is the secret of life?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["is_answerable"] is False
        assert "not available" in body["answer"].lower()
        assert body["evidence"] == []

    def test_short_question_returns_422(self, test_client):
        response = test_client.post("/api/v1/query", json={"question": "Hi"})
        assert response.status_code == 422

    def test_missing_question_returns_422(self, test_client):
        response = test_client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_evidence_has_required_fields(self, test_client, patched_query_pipeline):
        response = test_client.post(
            "/api/v1/query",
            json={"question": "What is Python?"},
        )
        body = response.json()
        evidence = body["evidence"][0]
        for field in ["document_id", "filename", "chunk_index", "score", "excerpt"]:
            assert field in evidence

    def test_reformulated_query_in_response(self, test_client, patched_query_pipeline):
        response = test_client.post(
            "/api/v1/query",
            json={"question": "Tell me about Python"},
        )
        body = response.json()
        assert body["reformulated_query"] is not None
        assert len(body["reformulated_query"]) > 0

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
