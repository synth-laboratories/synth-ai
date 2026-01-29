"""Backward-compatible URL helpers re-export."""

from __future__ import annotations

from synth_ai.core.utils.urls import (  # noqa: F401
    BACKEND_URL_API,
    BACKEND_URL_BASE,
    BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC,
    BACKEND_URL_SYNTH_RESEARCH_BASE,
    BACKEND_URL_SYNTH_RESEARCH_OPENAI,
    FRONTEND_URL_BASE,
    backend_demo_keys_url,
    backend_health_url,
    backend_me_url,
    join_url,
    local_backend_url,
    normalize_backend_base,
    normalize_base_url,
    normalize_inference_base,
)

__all__ = [
    "BACKEND_URL_BASE",
    "BACKEND_URL_API",
    "BACKEND_URL_SYNTH_RESEARCH_BASE",
    "BACKEND_URL_SYNTH_RESEARCH_OPENAI",
    "BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC",
    "FRONTEND_URL_BASE",
    "join_url",
    "normalize_base_url",
    "normalize_backend_base",
    "normalize_inference_base",
    "local_backend_url",
    "backend_health_url",
    "backend_me_url",
    "backend_demo_keys_url",
]
