"""Coding agent integrations (Claude Code, Codex, OpenCode).

These commands launch coding agents with Synth backend configuration
and optional session management for spending limits.
"""

from synth_ai.cli.agents.claude import claude_cmd
from synth_ai.cli.agents.codex import codex_cmd
from synth_ai.cli.agents.opencode import opencode_cmd

__all__ = [
    "claude_cmd",
    "codex_cmd",
    "opencode_cmd",
]


