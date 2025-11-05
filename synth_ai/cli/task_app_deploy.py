"""Compatibility wrapper for task-app deploy command."""

from __future__ import annotations

from synth_ai.cli.deploy import deploy_cmd as deploy_command  # type: ignore[attr-defined]

__all__ = ["deploy_command"]
