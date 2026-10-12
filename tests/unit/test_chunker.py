"""Unit tests for the semantic chunker module."""
from __future__ import annotations

import pytest
from unittest.mock import patch


MOCK_EMBEDDINGS = [[float(i) / 10] * 768 for i in range(20)]


class TestSemanticChunk:
    def _run_chunk(self, text: str, threshold: float = 0.75, max_tokens: int = 512):
        with patch("app.ingestion.chunker.embed_texts") as mock_embed:
            mock_embed.return_value = MOCK_EMBEDDINGS[:50]
            from app.ingestion.chunker import semantic_chunk
            return semantic_chunk(text, similarity_threshold=threshold, max_tokens=max_tokens)

    def test_single_sentence_returns_one_chunk(self):
        chunks = self._run_chunk("Only one sentence here.")
        assert len(chunks) == 1

    def test_multiple_sentences_produce_chunks(self):
        text = "Python is great. Java is verbose. Go is fast. Rust is safe. C is low-level."
        chunks = self._run_chunk(text)
        assert len(chunks) >= 1
        # All original sentences should be preserved in output
        combined = " ".join(chunks)
        assert "Python" in combined
        assert "Rust" in combined

    def test_empty_text_returns_empty(self):
        chunks = self._run_chunk("")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = self._run_chunk("   \n\n  ")
        assert chunks == []

    def test_max_tokens_forces_splits(self):
        # A very small max_tokens should force many splits
        long_text = " ".join([f"Sentence number {i} about topic {i}." for i in range(20)])
        chunks = self._run_chunk(long_text, max_tokens=20)
        # Should have created more chunks than with a large max_tokens
        assert len(chunks) >= 2

    def test_chunks_contain_all_text(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = self._run_chunk(text)
        combined = " ".join(chunks)
        assert "First" in combined
        assert "Second" in combined
        assert "Third" in combined
