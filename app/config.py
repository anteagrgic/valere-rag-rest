from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB (Qdrant)
    qdrant_url: str = Field(default="http://qdrant:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = Field(default="newsgroups")

    # LLM provider: "openai" | "ollama" | "mock"
    llm_provider: str = Field(default="mock")
    openai_api_key: str | None = None
    openai_model: str = Field(default="gpt-4o-mini")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")

    # Ingestion defaults
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=100)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
