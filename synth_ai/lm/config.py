"""
Configuration management for LM providers.
Loads sensitive configuration from environment variables.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def should_use_cache() -> bool:
    """
    Check if caching should be enabled based on environment variable.

    Returns:
        bool: True if caching is enabled (default), False if explicitly disabled.

    Note:
        Caching is controlled by the USE_ZYK_CACHE environment variable.
        Set to "false", "0", or "no" to disable caching.
    """
    cache_env = os.getenv("USE_ZYK_CACHE", "true").lower()
    return cache_env not in ("false", "0", "no")


# List of models that are considered reasoning models
# These models typically have special handling for temperature and other parameters
reasoning_models = ["o1", "o3-mini", "o3", "o4-mini", "claude-3-7-sonnet-latest"]


@dataclass
class SynthConfig:
    """Synth API configuration loaded from environment."""

    base_url: str
    api_key: str
    timeout: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "SynthConfig":
        """Load configuration from environment variables."""
        # Support both SYNTH_ and MODAL_ prefixes for compatibility
        base_url = os.getenv("SYNTH_BASE_URL") or os.getenv("MODAL_BASE_URL")
        api_key = os.getenv("SYNTH_API_KEY") or os.getenv("MODAL_API_KEY")

        if not base_url:
            raise ValueError(
                "SYNTH_BASE_URL or MODAL_BASE_URL not found in environment. "
                "Please set it in your .env file:\n"
                "SYNTH_BASE_URL=<your-synth-url>"
            )

        if not api_key:
            raise ValueError(
                "SYNTH_API_KEY or MODAL_API_KEY not found in environment. "
                "Please set it in your .env file:\n"
                "SYNTH_API_KEY=<your-api-key>"
            )

        # Ensure base URL includes /v1 for OpenAI client compatibility
        # OpenAI client doesn't automatically add /v1, so we need to include it
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        return cls(
            base_url=base_url.rstrip("/"),  # Remove trailing slash if present
            api_key=api_key,
            timeout=float(os.getenv("SYNTH_TIMEOUT", "120")),
            max_retries=int(os.getenv("SYNTH_MAX_RETRIES", "3")),
        )

    def get_base_url_without_v1(self) -> str:
        """Get base URL without /v1 for direct API calls."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    def __repr__(self) -> str:
        """Safe representation without exposing sensitive data."""
        return f"SynthConfig(base_url={self.base_url}, api_key=*****, timeout={self.timeout})"


@dataclass
class OpenAIConfig:
    """OpenAI API configuration loaded from environment."""

    api_key: str

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file:\n"
                "OPENAI_API_KEY=<your-openai-api-key>"
            )

        return cls(api_key=api_key)

    def __repr__(self) -> str:
        """Safe representation without exposing sensitive data."""
        return "OpenAIConfig(api_key=*****)"
