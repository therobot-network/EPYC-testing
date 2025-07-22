"""
Application settings and configuration management.
"""

import os
import yaml
from functools import lru_cache
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


def load_ec2_config() -> Dict[str, Any]:
    """Load EC2-specific configuration from ec2.yaml file."""
    config_path = Path("configs/ec2.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


class Settings(BaseSettings):
    """Application settings."""
    
    def __init__(self, **kwargs):
        # Load EC2 config first
        super().__init__(**kwargs)
        self._ec2_config = load_ec2_config()
    
    # Server configuration - use EC2 config if available
    @property
    def host(self) -> str:
        return self._ec2_config.get('application', {}).get('host', self.default_host)
    
    @property 
    def port(self) -> int:
        return self._ec2_config.get('application', {}).get('port', self.default_port)
    
    @property
    def log_level(self) -> str:
        return self._ec2_config.get('application', {}).get('log_level', self.default_log_level)
    
    # Private attributes for default values
    default_host: str = Field(default="0.0.0.0", env="HOST")
    default_port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    default_log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CORS configuration - use EC2 config if available
    @property
    def allowed_origins(self) -> List[str]:
        ec2_origins = self._ec2_config.get('security', {}).get('cors_origins', [])
        if ec2_origins:
            return ec2_origins
        return self.default_allowed_origins
    
    default_allowed_origins: List[str] = Field(
        default=["*"], 
        env="ALLOWED_ORIGINS"
    )
    
    # Model configuration - use EC2 storage paths if available
    @property
    def model_path(self) -> str:
        return self._ec2_config.get('storage', {}).get('model_cache', self.default_model_path_base)
    
    default_model_path_base: str = Field(default="./models", env="MODEL_PATH")
    default_model_path: Optional[str] = Field(default=None, env="DEFAULT_MODEL_PATH")
    
    # Performance settings - use EC2 optimizations if available
    @property
    def max_models_in_memory(self) -> int:
        # Use EC2 memory constraints if available
        max_memory_gb = self._ec2_config.get('performance', {}).get('max_memory_gb')
        if max_memory_gb:
            # Conservative estimate: allow 1 model per GB of memory
            return max(1, max_memory_gb - 1)  # Reserve 1GB for system
        return self.default_max_models_in_memory
    
    default_max_models_in_memory: int = Field(default=3, env="MAX_MODELS_IN_MEMORY")
    
    @property
    def max_batch_size(self) -> int:
        return self._ec2_config.get('performance', {}).get('batch_size', self.default_max_batch_size)
    
    default_max_batch_size: int = Field(default=32, env="MAX_BATCH_SIZE")
    
    # AWS configuration - use EC2 region if available
    @property
    def aws_region(self) -> str:
        return self._ec2_config.get('ec2', {}).get('region', self.default_aws_region)
    
    default_aws_region: str = Field(default="us-west-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    
    # Performance configuration
    inference_timeout: int = Field(default=30, env="INFERENCE_TIMEOUT")
    
    # Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # EC2-specific properties
    @property
    def ec2_instance_id(self) -> Optional[str]:
        return self._ec2_config.get('ec2', {}).get('instance_id')
    
    @property
    def ec2_public_ip(self) -> Optional[str]:
        return self._ec2_config.get('ec2', {}).get('public_ip')
    
    @property
    def ec2_private_ip(self) -> Optional[str]:
        return self._ec2_config.get('ec2', {}).get('private_ip')
    
    @property
    def ec2_instance_type(self) -> Optional[str]:
        return self._ec2_config.get('ec2', {}).get('instance_type')
    
    @property
    def torch_threads(self) -> Optional[int]:
        return self._ec2_config.get('performance', {}).get('torch_threads')
    
    @property
    def cpu_cores(self) -> Optional[int]:
        return self._ec2_config.get('performance', {}).get('cpu_cores')
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 