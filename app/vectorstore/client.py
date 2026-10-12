from __future__ import annotations

import threading

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.utils.exceptions import VectorStoreError
from app.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_client_instance: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Return the global singleton Qdrant client.
    Connects to the running Qdrant instance configured in settings.
    Thread-safe lazy initialization.

    Raises:
        VectorStoreError: When Qdrant is unreachable.
    """
    global _client_instance
    if _client_instance is None:
        with _lock:
            if _client_instance is None:
                settings = get_settings()
                try:
                    if settings.qdrant_url:
                        logger.info("Connecting to Qdrant at URL: %s", settings.qdrant_url)
                        _client_instance = QdrantClient(
                            url=settings.qdrant_url,
                            api_key=settings.qdrant_api_key,
                        )
                    else:
                        logger.info(
                            "Connecting to Qdrant at %s:%s",
                            settings.qdrant_host,
                            settings.qdrant_port,
                        )
                        _client_instance = QdrantClient(
                            host=settings.qdrant_host,
                            port=settings.qdrant_port,
                            api_key=settings.qdrant_api_key,
                        )
                    # Quick health check
                    _client_instance.get_collections()
                    logger.info("Qdrant connection established successfully.")
                except Exception as exc:
                    _client_instance = None
                    raise VectorStoreError(
                        f"Cannot connect to Qdrant: {exc}"
                    ) from exc
    return _client_instance
