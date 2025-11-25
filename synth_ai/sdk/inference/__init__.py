"""Inference SDK - model inference via Synth.

This module provides the InferenceClient for running model inference
through the Synth backend.

Example:
    from synth_ai.sdk.inference import InferenceClient
    
    client = InferenceClient(base_url, api_key)
    response = await client.create_chat_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

from __future__ import annotations

# Re-export from existing location
from synth_ai.inference.client import InferenceClient

__all__ = [
    "InferenceClient",
]

