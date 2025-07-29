"""Configuration management for trace storage."""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DuckDBConfig(BaseModel):
    """Configuration for DuckDB storage backend."""
    class Config:
        extra = 'forbid'
    
    db_path: str = Field(default="synth_ai/traces/traces.duckdb", description="Path to DuckDB database file")
    batch_size: int = Field(default=1000, description="Default batch size for bulk operations")
    enable_analytics_views: bool = Field(default=True, description="Whether to create analytics views")
    transaction_size: int = Field(default=10000, description="Number of operations per transaction")
    
    # Performance tuning
    memory_limit: Optional[str] = Field(default=None, description="DuckDB memory limit (e.g., '4GB')")
    threads: Optional[int] = Field(default=None, description="Number of threads for DuckDB")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_queries: bool = Field(default=False, description="Whether to log all SQL queries")


class TraceStorageConfig(BaseModel):
    """Main configuration for trace storage system."""
    class Config:
        extra = 'forbid'
    
    # Storage backend selection
    backend: str = Field(default="duckdb", description="Storage backend to use")
    
    # Backend-specific configs
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    
    # General settings
    auto_commit: bool = Field(default=True, description="Whether to auto-commit transactions")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed operations")
    retry_delay: float = Field(default=1.0, description="Delay between retry attempts in seconds")
    
    @classmethod
    def from_env(cls) -> "TraceStorageConfig":
        """Load configuration from environment variables."""
        # Simple env var loading without pydantic_settings
        config_dict = {}
        
        # Check for backend override
        backend = os.environ.get("TRACE_STORAGE_BACKEND")
        if backend:
            config_dict["backend"] = backend
        
        # Check for DuckDB settings
        duckdb_config = {}
        db_path = os.environ.get("TRACE_STORAGE_DUCKDB__DB_PATH")
        if db_path:
            duckdb_config["db_path"] = db_path
        
        batch_size = os.environ.get("TRACE_STORAGE_DUCKDB__BATCH_SIZE")
        if batch_size:
            duckdb_config["batch_size"] = int(batch_size)
        
        if duckdb_config:
            config_dict["duckdb"] = DuckDBConfig(**duckdb_config)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TraceStorageConfig":
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


# Global configuration instance
_config: Optional[TraceStorageConfig] = None


def get_config() -> TraceStorageConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = TraceStorageConfig.from_env()
    return _config


def set_config(config: TraceStorageConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def configure(**kwargs) -> TraceStorageConfig:
    """Configure trace storage with keyword arguments."""
    config = TraceStorageConfig(**kwargs)
    set_config(config)
    return config