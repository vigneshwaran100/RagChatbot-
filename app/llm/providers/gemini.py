from __future__ import annotations

from app.config import get_settings
from app.utils.exceptions import LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Google Gemini GenerativeModel API.

    Raises:
        LLMProviderError: On API or config failure.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise LLMProviderError("GOOGLE_API_KEY is not configured.")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise LLMProviderError("google-generativeai package not installed.") from exc

    try:
        genai.configure(api_key=settings.google_api_key)
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system_prompt,
        )
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = model.generate_content(
            user_prompt,
            generation_config=generation_config,
        )
        content = response.text
        logger.debug("Gemini response received (%d chars).", len(content))
        return content
    except Exception as exc:
        raise LLMProviderError(f"Gemini API error: {exc}") from exc
