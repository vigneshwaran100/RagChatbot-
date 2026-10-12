from __future__ import annotations

from app.config import get_settings
from app.utils.exceptions import LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def call_claude(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Anthropic Claude Messages API.

    Raises:
        LLMProviderError: On API or config failure.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise LLMProviderError("ANTHROPIC_API_KEY is not configured.")

    try:
        import anthropic
    except ImportError as exc:
        raise LLMProviderError("anthropic package not installed.") from exc

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        content = message.content[0].text
        logger.debug("Claude response received (%d chars).", len(content))
        return content
    except Exception as exc:
        raise LLMProviderError(f"Claude API error: {exc}") from exc
