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
from synth_ai.sdk.research.planning_reads import (
    AsyncPlanningReads,
    PlanningGraphRead,
    PlanningReadPosition,
    PlanningReads,
    PlanningTaskPage,
    PlanningTaskRead,
)

AsyncResearchClient = AsyncClient

__all__ = [
    "AsyncClient",
    "AsyncResearchClient",
    "AsyncOwnerReadClient",
    "AsyncPlanningReads",
    "Client",
    "DataPoolReadReference",
    "DatasetRevisionReadReference",
    "ProjectDataPoolDescriptor",
    "ProjectDataPoolInventory",
    "OwnerReadAccess",
    "OwnerReadClient",
    "OwnerReadScope",
    "PlanningGraphRead",
    "PlanningReadPosition",
    "PlanningReads",
    "PlanningTaskPage",
    "PlanningTaskRead",
    "RESEARCH_OPERATIONS",
    "ResearchClient",
    "research_operation",
]
