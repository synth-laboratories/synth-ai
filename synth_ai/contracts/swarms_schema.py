"""Deterministic single-model Swarms contract export; backend adoption pending."""

from __future__ import annotations

import json

from pydantic import BaseModel

from synth_ai.sdk.core.artifact_types import ArtifactLabelsUpdate
from synth_ai.sdk.swarms.types.approvals import (
    Approval,
    ApprovalDecisionReceipt,
    ApprovalDecisionRequest,
    ApprovalList,
)
from synth_ai.sdk.swarms.types.files import File, FileManifest, FileUploadReceipt, FileUploadStatus
from synth_ai.sdk.swarms.types.launch import (
    FileRef,
    Inputs,
    LaunchRequest,
    Limits,
    Models,
    Project,
    Providers,
    RepoSelector,
    ResolvedModels,
    ResolvedProviders,
)
from synth_ai.sdk.swarms.types.objectives import (
    Guidance,
    GuidanceCoverage,
    ObjectiveAnswer,
    ObjectiveResult,
    ObjectiveVerdict,
)
from synth_ai.sdk.swarms.types.outputs import (
    Output,
    OutputPage,
    OutputRef,
    OutputVerification,
    OutputVerificationError,
    TextContent,
    TransferTiming,
)
from synth_ai.sdk.swarms.types.programs import (
    ProgramChoice,
    ProgramFamily,
    ProgramFamilyList,
    ProgramRunView,
)
from synth_ai.sdk.swarms.types.projects import ProjectCreateRequest
from synth_ai.sdk.swarms.types.runs import (
    LaunchPreflight,
    ResolvedConfiguration,
    RunSnapshot,
    SwarmEvent,
    SwarmEventPage,
    SwarmResult,
    TaskOutcome,
)
from synth_ai.types.base import EventCursor, Page
from synth_ai.types.receipts import ControlReceipt
from synth_ai.types.response_values import response_schema

from .operations import PREFIX
from .swarms import ALL_OPERATIONS

MODELS: tuple[type[BaseModel], ...] = (
    Guidance,
    ObjectiveResult,
    ObjectiveVerdict,
    ObjectiveAnswer,
    GuidanceCoverage,
    ArtifactLabelsUpdate,
    ControlReceipt,
    ProgramRunView,
    ProgramChoice,
    ProgramFamily,
    ProgramFamilyList,
    Approval,
    ApprovalList,
    ApprovalDecisionRequest,
    ApprovalDecisionReceipt,
    FileRef,
    File,
    FileManifest,
    FileUploadReceipt,
    FileUploadStatus,
    Output,
    OutputRef,
    OutputVerification,
    OutputVerificationError,
    OutputPage,
    TextContent,
    TransferTiming,
    Page[File],
    Page[Output],
    Inputs,
    LaunchRequest,
    Limits,
    Models,
    Project,
    Providers,
    RepoSelector,
    ResolvedModels,
    ResolvedProviders,
    LaunchPreflight,
    ResolvedConfiguration,
    RunSnapshot,
    SwarmEvent,
    SwarmEventPage,
    SwarmResult,
    TaskOutcome,
    Page[RunSnapshot],
    Page[SwarmEvent, EventCursor],
    Page[TaskOutcome],
    ProjectCreateRequest,
)


def export_contract() -> str:
    return (
        json.dumps(
            {
                "schema_version": "swarms.v1",
                "status": "sdk_first_pending_backend",
                "models": {model.__name__: response_schema(model) for model in MODELS},
                "operations": [
                    {
                        "id": op.id,
                        "method": op.method,
                        "path": PREFIX + op.path,
                        "keyed": op.keyed,
                        "mutation": op.mutation,
                        "required_scope": op.required_scope,
                        "may_charge": op.may_charge,
                        "idempotent": op.idempotent,
                    }
                    for op in ALL_OPERATIONS
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    print(export_contract(), end="")
