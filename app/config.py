from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── Qdrant ───────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "rag_knowledge_base"
    qdrant_api_key: str | None = None
    qdrant_url: str | None = None           # overrides host+port when set

    # ── Embedding ────────────────────────────────
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: str = "cpu"           # cpu | cuda | mps
    embedding_batch_size: int = 32
    embedding_dimension: int = 768          # all-mpnet-base-v2 output dim

    # ── Chunking ─────────────────────────────────
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    semantic_similarity_threshold: float = 0.75

    # ── LLM — Groq (primary) ─────────────────────
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # ── LLM — OpenAI (fallback 1) ────────────────
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # ── LLM — Anthropic Claude (fallback 2) ──────
    anthropic_api_key: str | None = None
    claude_model: str = "claude-3-haiku-20240307"

    # ── LLM — Google Gemini (fallback 3) ─────────
    google_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    # ── Retrieval ────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.4

    # ── Reliability ──────────────────────────────
    confidence_threshold: float = 0.4

    # ── Upload ───────────────────────────────────
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = ["pdf", "txt", "md"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
