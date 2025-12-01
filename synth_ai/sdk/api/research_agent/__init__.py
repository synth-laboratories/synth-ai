"""Research Agent API for running AI-assisted code analysis and optimization.

This module provides both CLI and SDK interfaces for research agent jobs.

CLI Usage:
    uvx synth-ai agent run --config my_config.toml --poll

SDK Usage:
    from synth_ai.sdk.api.research_agent import (
        ResearchAgentJob,
        ResearchAgentJobConfig,
        ResearchConfig,
        DatasetSource,
        OptimizationTool,
        MIPROConfig,
        GEPAConfig,
    )

    # Create typed config
    research_config = ResearchConfig(
        task_description="Optimize prompt for banking classification",
        tools=[OptimizationTool.MIPRO],
        datasets=[
            DatasetSource(
                source_type="huggingface",
                hf_repo_id="PolyAI/banking77",
            )
        ],
    )

    job_config = ResearchAgentJobConfig(
        research=research_config,
        repo_url="https://github.com/my-org/my-pipeline",
        model="gpt-5.1-codex-mini",
        max_agent_spend_usd=25.0,
    )

    job = ResearchAgentJob(config=job_config)
    job_id = job.submit()
    result = job.poll_until_complete()
"""

from __future__ import annotations

from typing import Any

from .config import (
    DatasetSource,
    GEPAConfig,
    MIPROConfig,
    ModelProvider,
    OptimizationTool,
    PermittedModel,
    PermittedModelsConfig,
    ResearchConfig,
)
from .job import (
    ResearchAgentJob,
    ResearchAgentJobConfig,
    ResearchAgentJobPoller,
)

__all__ = [
    # CLI
    "register",
    # SDK - Main classes
    "ResearchAgentJob",
    "ResearchAgentJobConfig",
    "ResearchAgentJobPoller",
    # SDK - Config types
    "ResearchConfig",
    "DatasetSource",
    "OptimizationTool",
    "MIPROConfig",
    "GEPAConfig",
    "PermittedModelsConfig",
    "PermittedModel",
    "ModelProvider",
]


def register(cli: Any) -> None:
    """Register the agent command with the CLI."""
    from .cli import register as _register

    _register(cli)
