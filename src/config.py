from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System-wide configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    VLLM_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_MODEL: str = "Qwen/Qwen2.5-14B-Instruct"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "muhawir_docs"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    POSTGRES_DSN: str = "postgresql+asyncpg://postgres:changeme@localhost:5432/muhawir"
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    API_KEY: str = ""
    FASTTEXT_MODEL_PATH: str = "./models/lid.176.bin"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL: int = 3600
    FAITHFULNESS_THRESHOLD: float = 0.8
    MAX_RETRIES: int = 2
    RRF_K: int = 60

    # Multi-hop reasoning settings
    MAX_HOPS: int = 3
    COMPLEXITY_THRESHOLD: float = 0.6
    SUB_QUERY_MAX: int = 4
    HOP_CONFIDENCE_THRESHOLD: float = 0.7
    SYNTHESIS_MODEL: str = ""  # If empty, uses VLLM_MODEL


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
