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
    llm_provider: str = Field(default="ollama")
    openai_api_key: str | None = None
    openai_model: str = Field(default="gpt-4o-mini")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")

    # Ingestion defaults
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=100)

     # -------------------------
    # Advanced RAG knobs  (NOVO)
    # -------------------------
    # Globalne postavke koje koristi napredni pipeline ako klijent ne pošalje per-request opcije
    enable_query_translation: bool = Field(
        default=True,
        description="Ako je embedding ENG-only, drži uključeno; za multilingual možeš ugasiti."
    )
    enable_multi_query: bool = Field(default=True)
    enable_rag_fusion: bool = Field(default=True)
    multi_query_n: int = Field(default=4)
    rrf_k_smooth: int = Field(default=60)

    # LLM timeout (u sekundama)
    llm_timeout: int = Field(default=300, description="LLM HTTP timeout in seconds")
    # Pydantic v2 settings
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -------------------------
    # Alias/proxy svojstva  (NOVO)
    # -------------------------
    @property
    def llm_model(self) -> str:
        """
        Jedinstvena točka istine za naziv chat modela,
        bez obzira je li provider 'openai' ili 'ollama'.
        """
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "ollama":
            return self.ollama_model
        # mock ili drugi – vrati nešto benigno
        return "mock"

    @property
    def ollama_url(self) -> str:
        """
        Udoban alias; neka llm.py koristi ovo za base URL.
        """
        return self.ollama_base_url

settings = Settings()
