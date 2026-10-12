from __future__ import annotations

from app.config import get_settings
from app.utils.exceptions import LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Groq API and return the assistant message text.

    Raises:
        LLMProviderError: On API or config failure.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMProviderError("GROQ_API_KEY is not configured.")

    try:
        from groq import Groq
    except ImportError as exc:
        raise LLMProviderError("groq package not installed.") from exc

    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        logger.debug("Groq response received (%d chars).", len(content))
        return content
    except Exception as exc:
        raise LLMProviderError(f"Groq API error: {exc}") from exc
