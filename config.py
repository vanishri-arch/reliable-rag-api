from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Reliable RAG API"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: str = "data"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_relevance: float = 0.25
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
