"""
Application settings and configuration management.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CORS configuration
    allowed_origins: List[str] = Field(
        default=["*"], 
        env="ALLOWED_ORIGINS"
    )
    
    # Model configuration
    model_path: str = Field(default="./models", env="MODEL_PATH")
    default_model_path: Optional[str] = Field(default=None, env="DEFAULT_MODEL_PATH")
    max_models_in_memory: int = Field(default=3, env="MAX_MODELS_IN_MEMORY")
    
    # AWS configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    
    # Performance configuration
    max_batch_size: int = Field(default=32, env="MAX_BATCH_SIZE")
    inference_timeout: int = Field(default=30, env="INFERENCE_TIMEOUT")
    
    # Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 