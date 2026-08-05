"""Dependency-clean implementation of the public Synth Research SDK."""

from synth_ai.sdk.research.client import AsyncClient, Client
from synth_ai.sdk.research.facade import ResearchClient
from synth_ai.sdk.research.managed_inference import ManagedInferenceClient
from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS, research_operation

AsyncResearchClient = AsyncClient

__all__ = [
    "AsyncClient",
    "AsyncResearchClient",
    "Client",
    "ManagedInferenceClient",
    "RESEARCH_OPERATIONS",
    "ResearchClient",
    "research_operation",
]
