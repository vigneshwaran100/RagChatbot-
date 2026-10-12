from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.config import get_settings
from app.embeddings.mpnet import embed_texts
from app.ingestion.chunker import semantic_chunk
from app.ingestion.metadata import build_chunk_metadata
from app.ingestion.parser import parse_document
from app.schemas.ingest import IngestResponse
from app.utils.exceptions import (
    ChunkingError,
    DocumentParseError,
    EmbeddingError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.utils.logger import get_logger
from app.vectorstore.collections import ensure_collection
from app.vectorstore.indexer import upsert_chunks

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description=(
        "Upload a PDF, TXT, or Markdown file. The document is parsed with Docling, "
        "split into semantic chunks, embedded with MPNet, and stored in Qdrant."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, TXT, or MD file to ingest"),
    collection: str | None = None,
) -> IngestResponse:
    settings = get_settings()

    # ── Validate file extension ─────────────────────────────────────────────
    ext = Path(file.filename or "").suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.allowed_extensions}",
        )

    # ── Validate file size ──────────────────────────────────────────────────
    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    filename = file.filename or f"upload_{uuid.uuid4()}.{ext}"
    document_id = str(uuid.uuid4())
    logger.info("Ingesting document '%s' (id=%s, size=%d bytes)", filename, document_id, len(file_bytes))

    # ── Parse ───────────────────────────────────────────────────────────────
    try:
        markdown_text = parse_document(file_bytes, filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))
    except DocumentParseError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # ── Semantic chunk ──────────────────────────────────────────────────────
    try:
        chunks = semantic_chunk(markdown_text)
    except ChunkingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document produced no usable text chunks.",
        )

    total_chunks = len(chunks)
    logger.info("Document '%s' split into %d chunks.", filename, total_chunks)

    # ── Embed ───────────────────────────────────────────────────────────────
    try:
        embeddings = embed_texts(chunks)
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # ── Build metadata ──────────────────────────────────────────────────────
    metadata_list = [
        build_chunk_metadata(
            document_id=document_id,
            filename=filename,
            chunk_index=i,
            total_chunks=total_chunks,
            chunk_text=chunk,
            file_type=ext,
        )
        for i, chunk in enumerate(chunks)
    ]

    # ── Ensure collection + upsert ──────────────────────────────────────────
    try:
        col_name = ensure_collection(collection)
        upsert_chunks(col_name, chunks, embeddings, metadata_list)
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        file_type=ext,
        total_chunks=total_chunks,
        collection=col_name,
    )
