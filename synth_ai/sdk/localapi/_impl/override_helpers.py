"""Agent-aware context override applicator for task apps.

This module provides helpers for task apps to apply context overrides
(AGENTS.md, skills, preflight scripts, env vars) sent by GEPA.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path

import synth_ai_py

from synth_ai.data.artifacts import ContextOverride, ContextOverrideStatus


class AgentType(str, Enum):
    """Supported agent types for context overrides."""

    CODEX = "codex"
    OPENCODE = "opencode"


def get_agent_skills_path(agent: str | AgentType, global_: bool = False) -> str:
    """Get the canonical skills file path for an agent."""

    agent_str = agent.value if isinstance(agent, AgentType) else str(agent)
    return synth_ai_py.localapi_get_agent_skills_path(agent_str, global_)


async def apply_context_overrides(
    overrides: list[ContextOverride] | None,
    workspace_dir: Path | str,
    agent: str | AgentType = AgentType.OPENCODE,
    allow_global: bool = False,
    override_bundle_id: str | None = None,
) -> list[ContextOverrideStatus]:
    """Apply context overrides to a workspace."""

    _ = agent  # agent-specific path selection handled by caller when needed
    overrides = overrides or []
    workspace_dir = str(workspace_dir)
    results = await asyncio.to_thread(
        synth_ai_py.localapi_apply_context_overrides,
        overrides,
        workspace_dir,
        allow_global,
        override_bundle_id,
    )
    return results


def get_applied_env_vars(overrides: list[ContextOverride] | None) -> dict[str, str]:
    """Extract all env vars from overrides for injection into agent process."""

    return synth_ai_py.localapi_get_applied_env_vars(overrides or [])


__all__ = [
    "AgentType",
    "get_agent_skills_path",
    "apply_context_overrides",
    "get_applied_env_vars",
]
