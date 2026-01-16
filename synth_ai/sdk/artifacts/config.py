"""Configuration utilities for artifacts CLI."""

import os
from dataclasses import dataclass

import click

from synth_ai.core.urls import synth_api_url

DEFAULT_TIMEOUT = 30.0


@dataclass
class ArtifactsConfig:
    """Configuration for artifacts CLI commands."""

    synth_base_url: str | None
    synth_user_key: str
    timeout: float = DEFAULT_TIMEOUT

    @property
    def api_base_url(self) -> str:
        """Get the API base URL (ensures /api suffix)."""
        return synth_api_url("", self.synth_base_url)


def resolve_backend_config(
    *,
    synth_user_key: str | None = None,
    timeout: float | None = None,
    synth_base_url: str | None = None,
) -> ArtifactsConfig:
    """Resolve backend configuration from environment or explicit values."""
    if synth_user_key is None:
        synth_user_key = os.getenv("SYNTH_API_KEY", "")
        if not synth_user_key:
            raise click.ClickException(
                "API key required. Set SYNTH_API_KEY env var or use --api-key option."
            )

    return ArtifactsConfig(
        synth_base_url=synth_base_url,
        synth_user_key=synth_user_key,
        timeout=timeout or DEFAULT_TIMEOUT,
    )
