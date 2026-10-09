"""Dependency-clean implementation of the public Synth Research SDK."""

from synth_ai.sdk.research.client import AsyncClient, Client
from synth_ai.sdk.research.contracts.resource_reads import (
    DataPoolReadReference,
    DatasetRevisionReadReference,
    ProjectDataPoolDescriptor,
    ProjectDataPoolInventory,
)
from synth_ai.sdk.research.facade import ResearchClient
from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS, research_operation
from synth_ai.sdk.research.owner_reads import (
    AsyncOwnerReadClient,
    OwnerReadAccess,
    OwnerReadClient,
    OwnerReadScope,
)

AsyncResearchClient = AsyncClient

__all__ = [
    "AsyncClient",
    "AsyncResearchClient",
    "AsyncOwnerReadClient",
    "Client",
    "DataPoolReadReference",
    "DatasetRevisionReadReference",
    "ProjectDataPoolDescriptor",
    "ProjectDataPoolInventory",
    "OwnerReadAccess",
    "OwnerReadClient",
    "OwnerReadScope",
    "RESEARCH_OPERATIONS",
    "ResearchClient",
    "research_operation",
]
