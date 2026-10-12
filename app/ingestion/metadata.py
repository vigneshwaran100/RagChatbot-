from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_chunk_metadata(
    document_id: str,
    filename: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    file_type: str,
    page_number: int | None = None,
) -> dict:
    """
    Build a standardised metadata payload for a single chunk.

    This dict is stored as the Qdrant point payload alongside the
    embedding vector. Every field here is available for filtering
    and is returned as evidence to the end user.

    Args:
        document_id:   UUID of the parent document.
        filename:      Original upload filename.
        chunk_index:   Zero-based index of this chunk within the document.
        total_chunks:  Total number of chunks the document was split into.
        chunk_text:    Raw text of the chunk (stored for excerpt display).
        file_type:     File extension without dot ('pdf', 'txt', 'md').
        page_number:   Originating page number if known (PDFs via Docling).

    Returns:
        Flat dict suitable for Qdrant payload storage.
    """
    return {
        # Identity
        "document_id": document_id,
        "filename": filename,
        "file_stem": Path(filename).stem,
        "file_type": file_type,
        # Position
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "page_number": page_number,
        # Content statistics
        "char_count": len(chunk_text),
        "approx_token_count": max(1, len(chunk_text) // 4),
        # Audit
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
