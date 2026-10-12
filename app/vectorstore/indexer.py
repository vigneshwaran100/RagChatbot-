from __future__ import annotations

import uuid

from qdrant_client.http import models as qdrant_models

from app.utils.exceptions import VectorStoreError
from app.utils.logger import get_logger
from app.vectorstore.client import get_qdrant_client

logger = get_logger(__name__)


def upsert_chunks(
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata_list: list[dict],
) -> int:
    """
    Upsert a batch of chunk vectors with their payloads into Qdrant.

    Args:
        collection_name: Target collection.
        chunks:          Raw text of each chunk.
        embeddings:      Corresponding embedding vectors.
        metadata_list:   Per-chunk metadata dicts (filename, chunk_index, …).

    Returns:
        Number of successfully upserted points.

    Raises:
        VectorStoreError: On Qdrant write failure.
    """
    if not chunks:
        return 0

    client = get_qdrant_client()
    points: list[qdrant_models.PointStruct] = []

    for chunk, vector, meta in zip(chunks, embeddings, metadata_list):
        payload = {**meta, "text": chunk}
        points.append(
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    try:
        client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info(
            "Upserted %d chunks into collection '%s'.",
            len(points),
            collection_name,
        )
        return len(points)
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to upsert {len(points)} chunks: {exc}"
        ) from exc


def delete_by_document_id(collection_name: str, document_id: str) -> int:
    """
    Delete all Qdrant points that belong to a given document_id.

    Args:
        collection_name: Target collection.
        document_id:     The document UUID to remove.

    Returns:
        Number of deleted points (approximate — Qdrant may batch internally).
    """
    client = get_qdrant_client()
    try:
        result = client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info(
            "Deleted chunks for document_id='%s' from '%s'.",
            document_id,
            collection_name,
        )
        # Qdrant returns operation_id; we fetch count separately
        return _count_by_document_id(collection_name, document_id)
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to delete document '{document_id}': {exc}"
        ) from exc


def _count_by_document_id(collection_name: str, document_id: str) -> int:
    """Count how many points belong to a document_id (used before deletion)."""
    client = get_qdrant_client()
    try:
        result = client.count(
            collection_name=collection_name,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        return result.count
    except Exception:
        return 0


def count_chunks_for_document(collection_name: str, document_id: str) -> int:
    """Public helper — count stored chunks for a given document_id."""
    return _count_by_document_id(collection_name, document_id)


def scroll_document_metadata(collection_name: str) -> list[dict]:
    """
    Scroll through all payloads and return one metadata record per
    unique document_id (for the /collections listing endpoint).
    """
    client = get_qdrant_client()
    seen: dict[str, dict] = {}
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            doc_id = rec.payload.get("document_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": rec.payload.get("filename", ""),
                    "file_type": rec.payload.get("file_type", ""),
                    "total_chunks": rec.payload.get("total_chunks", 0),
                    "ingested_at": rec.payload.get("ingested_at", ""),
                }
        if offset is None:
            break

    return list(seen.values())
