from __future__ import annotations

from qdrant_client.http import models as qdrant_models

from app.config import get_settings
from app.embeddings.mpnet import embed_query
from app.utils.exceptions import RetrievalError
from app.utils.logger import get_logger
from app.vectorstore.client import get_qdrant_client

logger = get_logger(__name__)


def search_vectors(
    query: str,
    collection_name: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Embed the query with MPNet and perform cosine similarity search in Qdrant.

    Args:
        query:           Search query string (already reformulated).
        collection_name: Target Qdrant collection.
        top_k:           Number of results to return.
        score_threshold: Minimum cosine similarity score filter.

    Returns:
        List of result dicts containing payload + score fields.
        Sorted descending by score.

    Raises:
        RetrievalError: On embedding or Qdrant failure.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

    try:
        logger.debug("Embedding query for vector search: '%s'", query[:80])
        vector = embed_query(query)
    except Exception as exc:
        raise RetrievalError(f"Failed to embed query: {exc}") from exc

    try:
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        raise RetrievalError(
            f"Qdrant search failed on collection '{collection_name}': {exc}"
        ) from exc

    results = []
    for hit in hits:
        payload = dict(hit.payload) if hit.payload else {}
        payload["_score"] = hit.score
        results.append(payload)

    logger.info(
        "Retrieved %d chunks (top_k=%d, threshold=%.2f) for query: '%s'",
        len(results),
        k,
        threshold,
        query[:60],
    )
    return results
