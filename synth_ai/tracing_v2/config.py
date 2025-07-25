"""
Configuration for provider model mappings.
Maintains lists of known models for each supported provider.
"""
import os

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