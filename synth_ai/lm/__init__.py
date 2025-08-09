"""
Synth AI Language Model Interface.

Provides a unified interface for multiple LLM providers including OpenAI and Synth.
"""

from .config import SynthConfig, OpenAIConfig
from .warmup import warmup_synth_model, get_warmup_status
from .unified_interface import (
    UnifiedLMProvider,
    OpenAIProvider,
    SynthProvider,
    UnifiedLMClient,
    create_provider,
)
from .vendors.synth_client import (
    AsyncSynthClient,
    SyncSynthClient,
    create_async_client,
    create_sync_client,
    create_chat_completion_async,
    create_chat_completion_sync,
)
from .core.main_v3 import LM

__all__ = [
    # Configuration
    "SynthConfig",
    "OpenAIConfig",
    # Warmup utilities
    "warmup_synth_model",
    "get_warmup_status",
    # Unified interface
    "UnifiedLMProvider",
    "OpenAIProvider",
    "SynthProvider",
    "UnifiedLMClient",
    "create_provider",
    # Synth client
    "AsyncSynthClient",
    "SyncSynthClient",
    "create_async_client",
    "create_sync_client",
    "create_chat_completion_async",
    "create_chat_completion_sync",
    # Core LM class
    "LM",
]

# Version info
__version__ = "0.1.0"
