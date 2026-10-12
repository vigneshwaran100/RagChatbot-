from __future__ import annotations

from functools import lru_cache

from app.embeddings.mpnet import get_embedding_model
from app.vectorstore.client import get_qdrant_client


def get_embedder():
    """FastAPI dependency: return the singleton embedding model."""
    return get_embedding_model()


def get_vector_store():
    """FastAPI dependency: return the singleton Qdrant client."""
    return get_qdrant_client()
