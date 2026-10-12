from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.utils.exceptions import EmbeddingError
from app.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_lock = threading.Lock()
_model_instance: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Return the global singleton MPNet embedding model.
    Thread-safe lazy initialization.
    """
    global _model_instance
    if _model_instance is None:
        with _lock:
            if _model_instance is None:
                settings = get_settings()
                logger.info(
                    "Loading embedding model: %s on device=%s",
                    settings.embedding_model,
                    settings.embedding_device,
                )
                _model_instance = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device,
                )
                logger.info("Embedding model loaded successfully.")
    return _model_instance


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings and return a list of float vectors.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats).

    Raises:
        EmbeddingError: On any model inference failure.
    """
    if not texts:
        return []
    try:
        settings = get_settings()
        model = get_embedding_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,   # cosine similarity == dot product
            show_progress_bar=False,
        )
        return embeddings.tolist()
    except Exception as exc:
        raise EmbeddingError(f"Failed to embed {len(texts)} texts: {exc}") from exc


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The search query.

    Returns:
        Single embedding vector.
    """
    return embed_texts([query])[0]
