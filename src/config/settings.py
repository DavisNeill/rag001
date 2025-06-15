"""Configuration globale du projet"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Configuration de l'application"""
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None # <-- AJOUTEZ CETTE LIGNE
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama2"
    embedding_model: str = "nomic-embed-text"
    
    # Database
    database_url: Optional[str] = None
    mongodb_url: Optional[str] = None
    
    # Vector Store
    chroma_persist_directory: str = "./data/vector_db"
    qdrant_url: Optional[str] = None
    
    # Application
    debug: bool = True
    log_level: str = "INFO"
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
