from __future__ import annotations

import re

import numpy as np

from app.config import get_settings
from app.embeddings.mpnet import embed_texts
from app.utils.exceptions import ChunkingError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Simple sentence boundary regex (handles ., !, ? followed by space or newline)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    raw = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _count_tokens(text: str) -> int:
    """
    Approximate token count (1 token ≈ 4 chars).
    Avoids importing tiktoken as a hard dependency.
    """
    return max(1, len(text) // 4)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def semantic_chunk(
    markdown_text: str,
    similarity_threshold: float | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """
    Split Markdown text into semantically coherent chunks.

    Algorithm:
    1. Split into sentences.
    2. Embed all sentences with MPNet.
    3. Walk through sentences; start a new chunk whenever:
       - cosine similarity between adjacent sentences drops below threshold, OR
       - the current chunk exceeds max_tokens.
    4. Return list of chunk strings.

    Args:
        markdown_text:        Full document Markdown.
        similarity_threshold: Override setting value.
        max_tokens:           Override setting value.

    Returns:
        List of chunk strings (non-empty).

    Raises:
        ChunkingError: On any processing failure.
    """
    settings = get_settings()
    threshold = similarity_threshold or settings.semantic_similarity_threshold
    max_tok = max_tokens or settings.chunk_max_tokens

    try:
        sentences = _split_sentences(markdown_text)
        if not sentences:
            logger.warning("No sentences extracted from document; returning full text as one chunk.")
            return [markdown_text] if markdown_text.strip() else []

        if len(sentences) == 1:
            return sentences

        logger.debug("Embedding %d sentences for semantic chunking.", len(sentences))
        embeddings = embed_texts(sentences)

        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]
        current_tokens: int = _count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            next_tokens = _count_tokens(sentences[i])

            # Start a new chunk if similarity drops OR token budget exceeded
            if sim < threshold or (current_tokens + next_tokens) > max_tok:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]
                current_tokens = next_tokens
            else:
                current_sentences.append(sentences[i])
                current_tokens += next_tokens

        # Flush the last chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        logger.info(
            "Semantic chunking produced %d chunks from %d sentences "
            "(threshold=%.2f, max_tokens=%d).",
            len(chunks),
            len(sentences),
            threshold,
            max_tok,
        )
        return chunks

    except ChunkingError:
        raise
    except Exception as exc:
        raise ChunkingError(f"Semantic chunking failed: {exc}") from exc
