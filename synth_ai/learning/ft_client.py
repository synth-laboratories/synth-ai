from __future__ import annotations

"""Backward-compatible shim for FtClient (moved to synth_ai.learning.sft.client)."""

from .sft.client import FtClient

__all__ = ["FtClient"]
