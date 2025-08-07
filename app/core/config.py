from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv("secrets/.env")

class StorageConfig(BaseModel):
    base_path: str = "./data"
    cache_duration: int = 86400
    max_file_size_mb: int = 50

    @validator("base_path")
    def validate_base_path(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

class ArxivConfig(BaseModel):
    api_base_url: str = "http://export.arxiv.org/api/query"
    pdf_base_url: str = "https://arxiv.org/pdf"
    rate_limit: float = 0.33
    timeout: int = 30
    max_results: int = 10

class AnthropicConfig(BaseModel):
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 4096
    temperature: float = 0.3
    section_chunk_size: int = 3000
    api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

class ProcessingConfig(BaseModel):
    max_pdf_size_mb: int = 50
    chunk_size: int = 1000
    max_concurrent_tasks: int = 5

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    cors_origins: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"

class Settings(BaseSettings):
    storage: StorageConfig = StorageConfig()
    arxiv: ArxivConfig = ArxivConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    processing: ProcessingConfig = ProcessingConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()
    
    app_name: str = "arXiv Document Processor"
    app_version: str = "1.0.0"
    debug: bool = False
    
    class Config:
        env_file = "secrets/.env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields like the flat ANTHROPIC_API_KEY

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "Settings":
        """Load settings from YAML configuration file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
                
            # Override with environment variables
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key and "anthropic" in config_data:
                config_data["anthropic"]["api_key"] = anthropic_key
            elif anthropic_key:
                config_data["anthropic"] = {"api_key": anthropic_key}
            
            return cls(**config_data)
        return cls()

    def get_storage_path(self, *paths: str) -> Path:
        """Get a path relative to the storage base path"""
        base = Path(self.storage.base_path)
        return base.joinpath(*paths)

    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.get_storage_path("papers"),
            self.get_storage_path("summaries"),
            self.get_storage_path("cache", "arxiv"),
            Path("logs"),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

_settings = None

def get_settings() -> Settings:
    """Get singleton settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
        _settings.ensure_directories()
    return _settings