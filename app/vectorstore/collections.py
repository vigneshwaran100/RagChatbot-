from __future__ import annotations

from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.utils.exceptions import CollectionNotFoundError, VectorStoreError
from app.utils.logger import get_logger
from app.vectorstore.client import get_qdrant_client

logger = get_logger(__name__)


def ensure_collection(collection_name: str | None = None) -> str:
    """
    Ensure the Qdrant collection exists; create it if it does not.

    Args:
        collection_name: Override the default collection name from settings.

    Returns:
        The resolved collection name.

    Raises:
        VectorStoreError: On any Qdrant communication failure.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()

    try:
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing:
            logger.info("Collection '%s' not found — creating.", name)
            client.create_collection(
                collection_name=name,
                vectors_config=qdrant_models.VectorParams(
                    size=settings.embedding_dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info("Collection '%s' created.", name)
        else:
            logger.debug("Collection '%s' already exists.", name)
    except Exception as exc:
        raise VectorStoreError(f"Failed to ensure collection '{name}': {exc}") from exc

    return name


def list_collections() -> list[str]:
    """Return names of all existing Qdrant collections."""
    client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]


def delete_collection(collection_name: str) -> bool:
    """
    Delete a Qdrant collection.

    Returns:
        True if deleted, False if it did not exist.
    """
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info("Collection '%s' deleted.", collection_name)
        return True
    except Exception:
        return False


def get_collection_info(collection_name: str) -> dict:
    """
    Retrieve info (vector count, status) for a collection.

    Raises:
        CollectionNotFoundError: If collection does not exist.
    """
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
    except Exception as exc:
        raise CollectionNotFoundError(
            f"Collection '{collection_name}' not found: {exc}"
        ) from exc
