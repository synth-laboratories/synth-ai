"""Backward-compatible shim for FtClient (moved to synth_ai.learning.sft.client)."""

from __future__ import annotations

from .sft.client import FtClient

__all__ = ["FtClient"]
