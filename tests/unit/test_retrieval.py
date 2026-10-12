"""Unit tests for the retrieval pipeline (searcher, reformulator, context)."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


class TestSearchVectors:
    def test_returns_list_of_dicts(self):
        mock_client = MagicMock()
        mock_hit = MagicMock()
        mock_hit.score = 0.9
        mock_hit.payload = {"filename": "test.txt", "text": "content", "chunk_index": 0}
        mock_client.search.return_value = [mock_hit]

        with (
            patch("app.retrieval.searcher.get_qdrant_client", return_value=mock_client),
            patch("app.retrieval.searcher.embed_query", return_value=[0.1] * 768),
        ):
            from app.retrieval.searcher import search_vectors
            results = search_vectors("test query", "test_collection")

        assert len(results) == 1
        assert results[0]["_score"] == 0.9
        assert results[0]["filename"] == "test.txt"

    def test_empty_results_when_no_hits(self):
        mock_client = MagicMock()
        mock_client.search.return_value = []

        with (
            patch("app.retrieval.searcher.get_qdrant_client", return_value=mock_client),
            patch("app.retrieval.searcher.embed_query", return_value=[0.1] * 768),
        ):
            from app.retrieval.searcher import search_vectors
            results = search_vectors("unknown query", "test_collection")

        assert results == []


class TestAssembleContext:
    def test_empty_results_returns_unanswerable(self):
        from app.retrieval.context import assemble_context
        evidence, answerable = assemble_context([])
        assert not answerable
        assert evidence == []

    def test_low_score_returns_unanswerable(self):
        with patch("app.retrieval.context.get_settings") as mock_cfg:
            mock_cfg.return_value.confidence_threshold = 0.7
            from app.retrieval.context import assemble_context
            results = [{"_score": 0.3, "filename": "f.txt", "text": "t", "chunk_index": 0, "document_id": "d", "page_number": None}]
            evidence, answerable = assemble_context(results)
        assert not answerable

    def test_high_score_returns_answerable(self):
        with patch("app.retrieval.context.get_settings") as mock_cfg:
            mock_cfg.return_value.confidence_threshold = 0.4
            from app.retrieval.context import assemble_context
            results = [{
                "_score": 0.9, "filename": "doc.txt", "text": "Python is great.",
                "chunk_index": 0, "document_id": "abc", "page_number": None,
            }]
            evidence, answerable = assemble_context(results)
        assert answerable
        assert len(evidence) == 1
        assert evidence[0].filename == "doc.txt"


class TestReformulateQuery:
    def test_valid_json_parsed_correctly(self):
        valid_response = json.dumps({
            "refined_query": "What is Python?",
            "search_terms": ["Python", "programming language"],
            "intent": "Learn about Python.",
            "requires_multiple_chunks": False,
        })

        with patch("app.retrieval.reformulator.call_llm", return_value=(valid_response, "groq")):
            from app.retrieval.reformulator import reformulate_query
            result = reformulate_query("Tell me about Python")

        assert result.refined_query == "What is Python?"
        assert "Python" in result.search_terms

    def test_invalid_json_raises_error(self):
        with patch("app.retrieval.reformulator.call_llm", return_value=("NOT JSON", "groq")):
            from app.retrieval.reformulator import reformulate_query
            from app.utils.exceptions import QueryReformulationError
            with pytest.raises(QueryReformulationError):
                reformulate_query("some question")
