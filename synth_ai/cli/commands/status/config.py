"""Configuration helpers for status commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackendConfig:
    base_url: str
    api_key: str | None
    timeout: float = 30.0
