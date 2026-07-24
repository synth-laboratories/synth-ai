"""Dependency-clean implementation of the public Synth Research SDK."""

from synth_ai.core.research.client import AsyncClient, Client
from synth_ai.core.research.facade import ResearchClient
from synth_ai.core.research.operations import RESEARCH_OPERATIONS, research_operation

AsyncResearchClient = AsyncClient

__all__ = [
    "AsyncClient",
    "AsyncResearchClient",
    "Client",
    "RESEARCH_OPERATIONS",
    "ResearchClient",
    "research_operation",
]
