"""Configuration utilities for artifacts CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass

import click

from synth_ai.core.env import get_backend_from_env

DEFAULT_TIMEOUT = 30.0


@dataclass
class ArtifactsConfig:
    """Configuration for artifacts CLI commands."""
    
    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    
    @property
    def api_base_url(self) -> str:
        """Get the API base URL (ensures /api suffix)."""
        base = self.base_url.rstrip("/")
        if not base.endswith("/api"):
            base = f"{base}/api"
        return base


def resolve_backend_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> ArtifactsConfig:
    """Resolve backend configuration from environment or explicit values."""
    if base_url is None:
        base, key = get_backend_from_env()
        base_url = base
        if api_key is None:
            api_key = key
    
    if api_key is None:
        api_key = os.getenv("SYNTH_API_KEY", "")
        if not api_key:
            raise click.ClickException(
                "API key required. Set SYNTH_API_KEY env var or use --api-key option."
            )
    
    return ArtifactsConfig(
        base_url=base_url or "",
        api_key=api_key,
        timeout=timeout or DEFAULT_TIMEOUT,
    )

