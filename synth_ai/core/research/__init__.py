"""Dependency-clean implementation of the public Synth Research SDK."""

from synth_ai.core.research.client import AsyncResearchClient, ResearchClient
from synth_ai.core.research.operations import RESEARCH_OPERATIONS, research_operation

__all__ = [
    "AsyncResearchClient",
    "RESEARCH_OPERATIONS",
    "ResearchClient",
    "research_operation",
]
