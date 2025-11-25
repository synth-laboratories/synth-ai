"""Research Agent SDK - scaffold tuning, evaluation, and trace analysis.

This module provides the SDK for running research agent jobs:
- Scaffold tuning
- Evaluation
- Trace analysis

Example:
    from synth_ai.sdk.research_agent import ResearchAgentJob
    
    job = ResearchAgentJob.from_config("config.toml")
    job.submit()
    result = job.poll_until_complete()
"""

from __future__ import annotations

# Re-export from existing location
from synth_ai.api.research_agent.job import (
    ResearchAgentJob,
    ResearchAgentJobConfig,
    ResearchAgentJobPoller,
    AlgorithmType,
    BackendType,
)

__all__ = [
    "ResearchAgentJob",
    "ResearchAgentJobConfig",
    "ResearchAgentJobPoller",
    "AlgorithmType",
    "BackendType",
]

