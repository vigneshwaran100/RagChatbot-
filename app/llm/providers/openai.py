from __future__ import annotations

from app.config import get_settings
from app.utils.exceptions import LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def call_openai(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call the OpenAI Chat Completions API.

    Raises:
        LLMProviderError: On API or config failure.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMProviderError("OPENAI_API_KEY is not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMProviderError("openai package not installed.") from exc

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        logger.debug("OpenAI response received (%d chars).", len(content))
        return content
    except Exception as exc:
        raise LLMProviderError(f"OpenAI API error: {exc}") from exc
