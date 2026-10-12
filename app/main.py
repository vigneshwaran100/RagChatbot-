from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.middleware import RequestLoggingMiddleware, add_cors
from app.api.routes import documents, ingest, query
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: pre-warm the embedding model and verify
    Qdrant connectivity before accepting any requests.
    """
    settings = get_settings()
    logger.info("Starting RAG Knowledge Assistant (env=%s)", settings.app_env)

    # Pre-warm embedding model (downloads on first run)
    try:
        from app.embeddings.mpnet import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model ready.")
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)

    # Verify Qdrant connection
    try:
        from app.vectorstore.client import get_qdrant_client
        from app.vectorstore.collections import ensure_collection
        get_qdrant_client()
        ensure_collection()
        logger.info("Qdrant connection verified.")
    except Exception as exc:
        logger.warning("Qdrant not available at startup: %s", exc)

    yield

    logger.info("RAG Knowledge Assistant shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Knowledge Assistant",
        description=(
            "AI-powered knowledge assistant that answers questions from a "
            "collection of documents with evidence and source citations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware
    add_cors(app)
    app.add_middleware(RequestLoggingMiddleware)

    # Routers
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Liveness probe — returns 200 when the app is running."""
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness probe — checks Qdrant connectivity.
        Returns 503 when Qdrant is unreachable.
        """
        try:
            from app.vectorstore.client import get_qdrant_client
            client = get_qdrant_client()
            client.get_collections()
            return {"status": "ready", "qdrant": "connected"}
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "qdrant": str(exc)},
            )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.app_host, port=s.app_port, reload=True)
