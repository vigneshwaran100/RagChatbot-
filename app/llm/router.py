from __future__ import annotations

from typing import Callable

from app.utils.exceptions import AllProvidersFailedError, LLMProviderError
from app.utils.logger import get_logger
from app.llm.providers.groq import call_groq
from app.llm.providers.openai import call_openai
from app.llm.providers.claude import call_claude
from app.llm.providers.gemini import call_gemini

logger = get_logger(__name__)

# Priority-ordered list of (provider_name, callable)
_PROVIDERS: list[tuple[str, Callable[..., str]]] = [
    ("groq", call_groq),
    ("openai", call_openai),
    ("claude", call_claude),
    ("gemini", call_gemini),
]


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[str, str]:
    """
    Call the first available LLM provider in priority order:
    Groq → OpenAI → Claude → Gemini.

    Args:
        system_prompt: System instruction for the LLM.
        user_prompt:   User message / question.
        temperature:   Sampling temperature (0.0 = deterministic).
        max_tokens:    Maximum output tokens.

    Returns:
        Tuple of (response_text, provider_name_used).

    Raises:
        AllProvidersFailedError: When every provider fails or is unconfigured.
    """
    errors: list[str] = []

    for name, fn in _PROVIDERS:
        try:
            logger.debug("Trying LLM provider: %s", name)
            result = fn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info("LLM call succeeded via provider '%s'.", name)
            return result, name
        except LLMProviderError as exc:
            logger.warning("Provider '%s' failed: %s", name, exc)
            errors.append(f"{name}: {exc}")

    raise AllProvidersFailedError(
        "All LLM providers failed.\n" + "\n".join(errors)
    )
