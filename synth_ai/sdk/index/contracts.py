"""Canonical Index boundary values; backend owns the public wire vocabulary.

See sibling docs/drafts/synth-index-contribution-format-2026-09-12.md.
Uploaded metadata expresses claims and release intent, never permission or QA.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=2048)]


class IndexContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ContributionKind(StrEnum):
    RESEARCH_REPORT = "research_report"
    DATASET = "dataset"
    RECIPE = "recipe"
    IMPLEMENTATION = "implementation"
    MODEL = "model"
    REPLICATION = "replication"
    ENGINEERING_NOTE = "engineering_note"


class ResearchArea(StrEnum):
    DATA = "data"
    ENVIRONMENT_DESIGN = "environment_design"
    EVALUATION = "evaluation"
    VERIFIER = "verifier"
    TRAINING_ALGORITHM = "training_algorithm"
    OPTIMIZATION_SEARCH = "optimization_search"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    AGENT_HARNESS = "agent_harness"
    MODEL = "model"
    SYSTEMS = "systems"


class WorkflowStage(StrEnum):
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    RESEARCH_OPERATIONS = "research_operations"


class ContributionOrigin(StrEnum):
    USER = "user"
    SYNTH = "synth"
    IMPORTED = "imported"


class ContributionAudience(StrEnum):
    PRIVATE = "private"
    ORG = "org"
    PUBLIC = "public"


class ContributionReference(IndexContract):
    contribution_id: Identifier
    revision_id: Identifier


def require_unique(values: tuple, field: str) -> None:
    """Reject repeated bounded identities rather than silently repairing evidence."""
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
