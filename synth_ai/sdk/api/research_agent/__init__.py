"""Research Agent API for running AI-assisted code analysis and optimization.

This module provides both CLI and SDK interfaces for research agent jobs.

CLI Usage:
    uvx synth-ai agent run --config my_config.toml --poll
    uvx synth-ai agent run --config my_config.toml --algorithm scaffold_tuning

SDK Usage:
    from synth_ai.sdk.api.research_agent import ResearchAgentJob

    # From config file
    job = ResearchAgentJob.from_config("my_config.toml")
    job.submit()
    result = job.poll_until_complete()

    # Programmatically
    job = ResearchAgentJob(
        algorithm="scaffold_tuning",
        repo_url="https://github.com/your-org/your-repo",
        backend="daytona",
        config={
            "objective": {"metric_name": "accuracy", "max_iterations": 5},
            "target_files": ["prompts/*.txt"],
        }
    )
    job.submit()
"""

from __future__ import annotations

from typing import Any

from .job import (
    ResearchAgentJob,
    ResearchAgentJobConfig,
    ResearchAgentJobPoller,
)

__all__ = [
    # CLI
    "register",
    # SDK
    "ResearchAgentJob",
    "ResearchAgentJobConfig",
    "ResearchAgentJobPoller",
]


def register(cli: Any) -> None:
    """Register the agent command with the CLI."""
    from .cli import register as _register

    _register(cli)
