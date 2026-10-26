"""Central environment configuration; no module loads model weights on import."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime limits and model choices, configurable with the RFP_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RFP_", env_file=".env", extra="ignore", populate_by_name=True, hide_input_in_errors=True
    )
    environment: Literal["development", "production"] = "development"
    api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GROQ_API_KEY")
    llm_model: str = "groq/openai/gpt-oss-120b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: Path = Path("models")
    model_threads: int = Field(1, ge=1, le=8)
    embedding_batch_size: int = Field(1, ge=1, le=64)
    embedding_max_tokens: int = Field(512, ge=128, le=512)
    dense_weight: float = Field(2.0, gt=0, le=10)
    reranker_weight: float = Field(0.25, ge=0, le=1)
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    public_base_url: str = "http://localhost:8000"
    input_root: Path = Path("bids")
    storage_dir: Path = Path("storage")
    chunk_words: int = Field(220, ge=80, le=1000)
    chunk_overlap: int = Field(35, ge=0)
    ocr_enabled: bool = True
    ocr_timeout_seconds: int = Field(60, ge=1)
    max_file_mb: int = Field(50, ge=1, le=500)
    max_files: int = Field(200, ge=1, le=5000)
    max_pdf_pages: int = Field(200, ge=1, le=5000)
    max_image_pixels: int = Field(4_000_000, ge=1000)
    max_active_jobs: int = Field(1, ge=1, le=8)
    max_search_concurrency: int = Field(1, ge=1, le=32)
    llm_timeout_seconds: int = Field(90, ge=1)
    task_timeout_seconds: int = Field(1800, ge=1)
    max_field_retries: int = Field(2, ge=0, le=3)
    log_level: str = "INFO"
    trace_content: bool = False

    @model_validator(mode="after")
    def validate_settings(self):
        """Reject unsafe production startup and impossible chunk settings."""
        if self.environment == "production" and len(self.api_key.get_secret_value()) < 24:
            raise ValueError("Production requires RFP_API_KEY with at least 24 characters")
        if self.chunk_overlap >= self.chunk_words:
            raise ValueError("chunk_overlap must be smaller than chunk_words")
        # Railway supplies the service domain, so A2A discovery needs no manual URL variable.
        if "public_base_url" not in self.model_fields_set and (domain := os.getenv("RAILWAY_PUBLIC_DOMAIN")):
            self.public_base_url = f"https://{domain.strip().rstrip('/')}"
        self.input_root = self.input_root.resolve()
        self.storage_dir = self.storage_dir.resolve()
        self.model_cache_dir = self.model_cache_dir.resolve()
        return self
