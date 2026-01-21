"""Authentication utilities for SDK and CLI."""

from synth_ai.core.auth.api_key import get_or_mint_synth_api_key
from synth_ai.core.auth.setup import run_setup

__all__ = [
    "get_or_mint_synth_api_key",
    "run_setup",
]
