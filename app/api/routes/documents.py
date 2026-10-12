from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.schemas.ingest import (
    DeleteDocumentResponse,
    DocumentInfo,
    ListDocumentsResponse,
)
from app.utils.exceptions import CollectionNotFoundError, VectorStoreError
from app.utils.logger import get_logger
from app.vectorstore.collections import get_collection_info, list_collections
from app.vectorstore.indexer import (
    count_chunks_for_document,
    delete_by_document_id,
    scroll_document_metadata,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)


@router.get(
    "",
    response_model=ListDocumentsResponse,
    summary="List all ingested documents",
)
async def list_documents(collection: str | None = None) -> ListDocumentsResponse:
    settings = get_settings()
    col_name = collection or settings.qdrant_collection_name

    try:
        info = get_collection_info(col_name)
    except CollectionNotFoundError:
        # Return empty list if collection doesn't exist yet
        return ListDocumentsResponse(
            collection=col_name, total_documents=0, documents=[]
        )
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    raw_docs = scroll_document_metadata(col_name)
    documents = [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            file_type=d["file_type"],
            total_chunks=d["total_chunks"],
            ingested_at=d["ingested_at"],
        )
        for d in raw_docs
    ]

    return ListDocumentsResponse(
        collection=col_name,
        total_documents=len(documents),
        documents=documents,
    )


@router.get(
    "/collections",
    summary="List all Qdrant collections",
)
async def list_all_collections() -> dict:
    try:
        cols = list_collections()
        return {"collections": cols, "total": len(cols)}
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document by ID",
)
async def delete_document(
    document_id: str,
    collection: str | None = None,
) -> DeleteDocumentResponse:
    settings = get_settings()
    col_name = collection or settings.qdrant_collection_name

    # Count before deleting for the response message
    chunk_count = count_chunks_for_document(col_name, document_id)
    if chunk_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in collection '{col_name}'.",
        )

    try:
        delete_by_document_id(col_name, document_id)
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return DeleteDocumentResponse(
        document_id=document_id,
        deleted_chunks=chunk_count,
    )
