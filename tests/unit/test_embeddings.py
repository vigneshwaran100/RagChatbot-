"""Unit tests for the MPNet embedding module."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestEmbedTexts:
    def test_empty_list_returns_empty(self):
        from app.embeddings.mpnet import embed_texts
        result = embed_texts([])
        assert result == []

    def test_returns_list_of_vectors(self):
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])

        with patch("app.embeddings.mpnet.get_embedding_model", return_value=mock_model):
            from app.embeddings.mpnet import embed_texts
            result = embed_texts(["Hello", "World"])

        assert len(result) == 2
        assert len(result[0]) == 768

    def test_vectors_are_lists_of_floats(self):
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([[0.5] * 768])

        with patch("app.embeddings.mpnet.get_embedding_model", return_value=mock_model):
            from app.embeddings.mpnet import embed_texts
            result = embed_texts(["test"])

        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    def test_embed_query_returns_single_vector(self):
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([[0.3] * 768])

        with patch("app.embeddings.mpnet.get_embedding_model", return_value=mock_model):
            from app.embeddings.mpnet import embed_query
            result = embed_query("what is machine learning?")

        assert isinstance(result, list)
        assert len(result) == 768

    def test_embedding_error_on_model_failure(self):
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("CUDA OOM")

        with patch("app.embeddings.mpnet.get_embedding_model", return_value=mock_model):
            from app.embeddings.mpnet import embed_texts
            from app.utils.exceptions import EmbeddingError
            with pytest.raises(EmbeddingError):
                embed_texts(["fail this"])
