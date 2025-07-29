"""Utility functions for storage operations."""
from .types import Provider


def detect_provider(model_name: str) -> Provider:
    """Detect provider from model name.
    
    Args:
        model_name: The model name to analyze
        
    Returns:
        The detected provider
    """
    if not model_name:
        return Provider.UNKNOWN
        
    model_lower = model_name.lower()
    
    if "gpt" in model_lower or "dall-e" in model_lower:
        return Provider.OPENAI
    elif "claude" in model_lower:
        return Provider.ANTHROPIC
    elif "gemini" in model_lower or "palm" in model_lower:
        return Provider.GOOGLE
    elif "azure" in model_lower:
        return Provider.AZURE
    elif any(local in model_lower for local in ["llama", "mistral", "mixtral", "phi"]):
        return Provider.LOCAL
    else:
        return Provider.UNKNOWN