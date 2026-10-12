from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.llm.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from app.llm.router import call_llm
from app.retrieval.context import assemble_context, format_context_for_llm
from app.retrieval.reformulator import reformulate_query
from app.retrieval.searcher import search_vectors
from app.schemas.query import QueryRequest, QueryResponse
from app.utils.exceptions import (
    AllProvidersFailedError,
    QueryReformulationError,
    RetrievalError,
    VectorStoreError,
)
from app.utils.logger import get_logger
from app.vectorstore.collections import ensure_collection

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)

_UNAVAILABLE_ANSWER = (
    "The information requested is not available in the provided documents."
)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Reformulates the question via LLM, performs semantic search in Qdrant, "
        "assembles grounded context, and generates a cited answer. "
        "Returns is_answerable=false with a clear message when documents lack information."
    ),
)
async def query_knowledge_base(body: QueryRequest) -> QueryResponse:
    settings = get_settings()
    col_name = body.collection or settings.qdrant_collection_name

    # ── Step 1: Query reformulation (DISABLED — using raw question directly) ──
    reformulated_query_str = body.question  # use original question as-is
    # try:
    #     reformulated = reformulate_query(body.question)
    #     reformulated_query_str = reformulated.refined_query
    #     logger.info("Reformulated query: '%s'", reformulated_query_str[:80])
    # except QueryReformulationError as exc:
    #     logger.warning("Query reformulation failed, using raw question: %s", exc)
    #     # Graceful degradation: continue with original question
    logger.info("Using raw question directly (reformulation disabled): '%s'", reformulated_query_str[:80])

    # ── Step 2: Ensure collection exists ────────────────────────────────────
    try:
        ensure_collection(col_name)
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    # ── Step 3: Vector search ────────────────────────────────────────────────
    try:
        search_results = search_vectors(
            query=reformulated_query_str,
            collection_name=col_name,
            top_k=body.top_k,
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    # ── Step 4: Assemble context and check answerability ─────────────────────
    evidence, is_answerable = assemble_context(search_results)

    if not is_answerable:
        return QueryResponse(
            question=body.question,
            answer=_UNAVAILABLE_ANSWER,
            is_answerable=False,
            evidence=[],
            reformulated_query=reformulated_query_str,
            provider_used="none",
        )

    # ── Step 5: Generate grounded answer ─────────────────────────────────────
    context_block = format_context_for_llm(evidence)
    user_prompt = ANSWER_USER_TEMPLATE.format(
        question=body.question,
        context=context_block,
    )

    try:
        answer, provider_used = call_llm(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
    except AllProvidersFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM unavailable: {exc}",
        )

    return QueryResponse(
        question=body.question,
        answer=answer.strip(),
        is_answerable=True,
        evidence=evidence,
        reformulated_query=reformulated_query_str,
        provider_used=provider_used,
    )
