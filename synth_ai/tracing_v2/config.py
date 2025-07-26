"""
Configuration for v3 tracing system and provider model mappings.

This module provides configuration options for the dual-mode tracing system
that supports both OpenTelemetry and v2 SessionTracer patterns, as well as
maintains lists of known models for each supported provider.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TracingConfig:
    """Configuration for the v3 tracing system."""
    
    # Tracing mode: "dual" (both OTel and v2), "otel" (OTel only), "v2" (v2 only), "disabled"
    mode: str = field(default_factory=lambda: os.getenv("SYNTH_TRACING_MODE", "dual"))
    
    # Sampling configuration (now relies on OTel SDK sampler)
    sample_rate: float = field(
        default_factory=lambda: float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
    )
    
    # Payload size limits
    max_payload_bytes: int = field(
        default_factory=lambda: int(os.getenv("SYNTH_MAX_PAYLOAD_BYTES", "10240"))
    )
    truncate_enabled: bool = field(
        default_factory=lambda: os.getenv("SYNTH_TRUNCATE_PAYLOADS", "true").lower() == "true"
    )
    
    # PII masking configuration
    mask_pii: bool = field(
        default_factory=lambda: os.getenv("SYNTH_MASK_PII", "true").lower() == "true"
    )
    
    # Flush configuration
    flush_interval_ms: int = field(
        default_factory=lambda: int(os.getenv("SYNTH_FLUSH_INTERVAL_MS", "5000"))
    )
    flush_on_exit: bool = field(
        default_factory=lambda: os.getenv("SYNTH_FLUSH_ON_EXIT", "true").lower() == "true"
    )
    export_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("SYNTH_EXPORT_TIMEOUT_MS", "30000"))
    )
    
    # OpenTelemetry configuration
    otel_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    )
    otel_headers: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    )
    otel_service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "synth-ai")
    )
    otel_service_version: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
    )
    otel_deployment_env: str = field(
        default_factory=lambda: os.getenv("DEPLOYMENT_ENV", os.getenv("OTEL_DEPLOYMENT_ENV", "development"))
    )
    otel_exporter: Optional[Any] = field(default=None)  # OTel SpanExporter instance
    
    # Langfuse configuration
    langfuse_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    
    # Performance tuning
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("SYNTH_TRACE_BATCH_SIZE", "100"))
    )
    max_queue_size: int = field(
        default_factory=lambda: int(os.getenv("SYNTH_MAX_QUEUE_SIZE", "10000"))
    )
    
    # Debug options
    debug: bool = field(
        default_factory=lambda: os.getenv("SYNTH_TRACE_DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("SYNTH_TRACE_LOG_LEVEL", "WARNING")
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode
        if self.mode not in ["dual", "otel", "v2", "disabled"]:
            raise ValueError(f"Invalid tracing mode: {self.mode}. Must be 'dual', 'otel', 'v2', or 'disabled'")
        
        # Validate sample rate
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {self.sample_rate}")
        
        # Validate payload size
        if self.max_payload_bytes < 1024:
            raise ValueError(f"Max payload bytes must be at least 1024, got {self.max_payload_bytes}")
    
    def is_tracing_enabled(self) -> bool:
        """Check if any tracing is enabled."""
        return self.mode != "disabled"
    
    def is_otel_enabled(self) -> bool:
        """Check if OpenTelemetry tracing is enabled."""
        return self.mode in ["dual", "otel"]
    
    def is_v2_enabled(self) -> bool:
        """Check if v2 SessionTracer is enabled."""
        return self.mode in ["dual", "v2"]
    
    def get_otel_headers(self) -> Dict[str, str]:
        """Parse OTEL headers into a dictionary."""
        if not self.otel_headers:
            return {}
        
        headers = {}
        for pair in self.otel_headers.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                headers[key.strip()] = value.strip()
        return headers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "max_payload_bytes": self.max_payload_bytes,
            "truncate_enabled": self.truncate_enabled,
            "flush_interval_ms": self.flush_interval_ms,
            "flush_on_exit": self.flush_on_exit,
            "otel_enabled": self.is_otel_enabled(),
            "v2_enabled": self.is_v2_enabled(),
            "langfuse_enabled": self.langfuse_enabled,
            "debug": self.debug,
            "log_level": self.log_level,
        }


# Global configuration instance
_config: Optional[TracingConfig] = None


def get_config() -> TracingConfig:
    """Get the global tracing configuration."""
    global _config
    if _config is None:
        _config = TracingConfig()
    return _config


def set_config(config: TracingConfig) -> None:
    """Set the global tracing configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = None


# Configuration shortcuts
def is_tracing_enabled() -> bool:
    """Check if any tracing is enabled."""
    config = get_config()
    return config.sample_rate > 0.0 and (config.is_otel_enabled() or config.is_v2_enabled())


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return get_config().debug


def get_sample_rate() -> float:
    """Get the current sample rate."""
    return get_config().sample_rate


def get_max_payload_bytes() -> int:
    """Get the maximum payload size in bytes."""
    return get_config().max_payload_bytes


# OpenAI models
OPENAI_MODELS = {
    # GPT-4 variants
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    
    # GPT-3.5 variants
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-instruct",
    
    # O-series models (reasoning models)
    "o1",
    "o1-preview",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4",  # Future-proofing
    
    # Legacy models
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "davinci",
    "curie",
    "babbage",
    "ada",
    
    # Embeddings
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    
    # Other
    "whisper-1",
    "tts-1",
    "tts-1-hd",
    "dall-e-2",
    "dall-e-3",
}

# Azure OpenAI models (same as OpenAI but may have custom deployment names)
AZURE_OPENAI_MODELS = OPENAI_MODELS.copy()

# Common Azure deployment name patterns
AZURE_DEPLOYMENT_PATTERNS = {
    "gpt-35-turbo",  # Azure often uses gpt-35 instead of gpt-3.5
    "gpt-4-32k",
    "gpt-4-turbo",
    "text-embedding-ada-002",
}

# Anthropic models
ANTHROPIC_MODELS = {
    # Claude 3 family
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    
    # Claude 2 family
    "claude-2.1",
    "claude-2.0",
    "claude-2",
    
    # Claude Instant
    "claude-instant-1.2",
    "claude-instant-1.1",
    "claude-instant-1.0",
    "claude-instant-1",
    
    # Legacy
    "claude-1.3",
    "claude-1.2",
    "claude-1.0",
    "claude-1",
    
    # Shortened versions that might appear
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for comparison."""
    if not model_name:
        return ""
    return model_name.lower().strip()


def is_openai_model(model_name: str) -> bool:
    """Check if the model name is an OpenAI model."""
    normalized = normalize_model_name(model_name)
    
    # Direct match
    if normalized in OPENAI_MODELS:
        return True
    
    # Check for o-series pattern (o1, o2, o3, etc.)
    if normalized.startswith("o") and len(normalized) >= 2:
        if normalized[1:].isdigit() or normalized.startswith(("o1-", "o3-", "o4-")):
            return True
    
    # Fallback to pattern matching
    if any(pattern in normalized for pattern in ["gpt-", "davinci", "curie", "babbage", "ada", "text-embedding", "whisper", "tts-", "dall-e"]):
        return True
    
    return False


def is_azure_openai_model(model_name: str, endpoint: str = "") -> bool:
    """Check if the model name is an Azure OpenAI model."""
    normalized = normalize_model_name(model_name)
    
    # If endpoint contains "azure", it's likely Azure OpenAI
    if endpoint and "azure" in endpoint.lower():
        # Could be any OpenAI model hosted on Azure
        return is_openai_model(model_name) or any(pattern in normalized for pattern in AZURE_DEPLOYMENT_PATTERNS)
    
    # Check Azure-specific naming patterns
    if "gpt-35" in normalized:  # Azure uses gpt-35 instead of gpt-3.5
        return True
    
    # Otherwise check if it's an OpenAI model (Azure hosts OpenAI models)
    return normalized in AZURE_OPENAI_MODELS or normalized in AZURE_DEPLOYMENT_PATTERNS


def is_anthropic_model(model_name: str) -> bool:
    """Check if the model name is an Anthropic model."""
    normalized = normalize_model_name(model_name)
    
    # Direct match
    if normalized in ANTHROPIC_MODELS:
        return True
    
    # Pattern matching for Claude models
    if "claude" in normalized:
        return True
    
    return False


def detect_provider(model_name: str, endpoint: str = "") -> str:
    """
    Detect the provider based on model name and endpoint.
    Returns: "openai", "azure_openai", "anthropic", or None
    """
    if not model_name:
        return None
    
    normalized = normalize_model_name(model_name)
    
    # Check Anthropic first (most specific)
    if is_anthropic_model(model_name):
        return "anthropic"
    
    # Check if it's an OpenAI model
    if is_openai_model(model_name):
        # If endpoint explicitly contains Azure, it's Azure OpenAI
        if endpoint and "azure" in endpoint.lower():
            return "azure_openai"
        # If it uses Azure-specific naming (gpt-35), it's Azure OpenAI
        if "gpt-35" in normalized:
            return "azure_openai"
        # Otherwise it's regular OpenAI
        return "openai"
    
    return None


# Storage Configuration
LOCAL_SYNTH = True  # If True, use local DuckDB storage. If False, use Synth Cloud (not yet implemented)

# DuckDB Configuration (for LOCAL_SYNTH mode)
DUCKDB_CONFIG = {
    "enabled": True if LOCAL_SYNTH else False,  # Auto-enable for local mode
    "db_path": "traces.duckdb",
    "batch_size": 1000,  # For batch uploads
    "auto_upload": True,  # Upload on session end
    "retention_days": 30,  # Data retention policy
}

# Synth Cloud Configuration (for future use)
SYNTH_CLOUD_CONFIG = {
    "enabled": not LOCAL_SYNTH,
    "api_url": "https://api.synth.ai",  # Placeholder
    "api_key": os.environ.get("SYNTH_API_KEY", ""),
}