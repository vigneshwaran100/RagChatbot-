from __future__ import annotations

from app.config import get_settings
from app.schemas.query import EvidenceItem
from app.utils.logger import get_logger

logger = get_logger(__name__)


def assemble_context(search_results: list[dict]) -> tuple[list[EvidenceItem], bool]:
    """
    Convert raw Qdrant search results into EvidenceItem objects and
    determine whether the retrieved context is sufficient to answer.

    Args:
        search_results: List of payload dicts from searcher.search_vectors(),
                        each containing metadata + '_score'.

    Returns:
        Tuple of:
          - evidence: list of EvidenceItem (empty if nothing retrieved)
          - is_answerable: True if best score >= confidence threshold
    """
    settings = get_settings()
    threshold = settings.confidence_threshold

    if not search_results:
        logger.info("No chunks retrieved — question is unanswerable.")
        return [], False

    best_score = search_results[0].get("_score", 0.0)
    is_answerable = best_score >= threshold

    if not is_answerable:
        logger.info(
            "Best retrieval score %.4f < threshold %.4f — marking as unanswerable.",
            best_score,
            threshold,
        )
        return [], False

    evidence: list[EvidenceItem] = []
    for result in search_results:
        evidence.append(
            EvidenceItem(
                document_id=result.get("document_id", ""),
                filename=result.get("filename", "unknown"),
                chunk_index=result.get("chunk_index", 0),
                score=round(result.get("_score", 0.0), 4),
                excerpt=result.get("text", "")[:1000],   # cap excerpt length
                page_number=result.get("page_number"),
            )
        )

    logger.debug(
        "Assembled %d evidence items (best_score=%.4f).", len(evidence), best_score
    )
    return evidence, True


def format_context_for_llm(evidence: list[EvidenceItem]) -> str:
    """
    Format evidence items into a numbered context block for the LLM prompt.

    Args:
        evidence: Validated EvidenceItem list.

    Returns:
        Multi-line string ready for insertion into the answer prompt.
    """
    parts: list[str] = []
    for i, item in enumerate(evidence, start=1):
        parts.append(
            f"[{i}] Source: {item.filename} (chunk {item.chunk_index}, "
            f"score={item.score})\n{item.excerpt}"
        )
    return "\n\n".join(parts)
