from __future__ import annotations

import json
import re

from app.llm.prompts import REFORMULATION_SYSTEM_PROMPT, REFORMULATION_USER_TEMPLATE
from app.llm.router import call_llm
from app.schemas.reformulation import ReformulatedQuery
from app.utils.exceptions import QueryReformulationError
from app.utils.logger import get_logger

logger = get_logger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(raw: str) -> str:
    """Strip markdown code fences if present, return raw JSON string."""
    match = _JSON_FENCE.search(raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


def reformulate_query(question: str) -> ReformulatedQuery:
    """
    Use the LLM to reformulate the user's question into a structured
    search object validated by the ReformulatedQuery Pydantic schema.

    Args:
        question: Raw user question string.

    Returns:
        ReformulatedQuery with refined_query, search_terms, intent.

    Raises:
        QueryReformulationError: On LLM failure or invalid JSON output.
    """
    user_prompt = REFORMULATION_USER_TEMPLATE.format(question=question)

    try:
        raw_response, provider = call_llm(
            system_prompt=REFORMULATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as exc:
        raise QueryReformulationError(
            f"LLM call for query reformulation failed: {exc}"
        ) from exc

    try:
        json_str = _extract_json(raw_response)
        data = json.loads(json_str)
        reformulated = ReformulatedQuery(**data)
        logger.info(
            "Query reformulated via '%s': '%s' → '%s'",
            provider,
            question[:60],
            reformulated.refined_query[:60],
        )
        return reformulated
    except Exception as exc:
        raise QueryReformulationError(
            f"Failed to parse reformulation JSON (provider={provider}): {exc}\n"
            f"Raw response: {raw_response[:300]}"
        ) from exc
