"""Deterministic single-model Swarms contract export; backend adoption pending."""

from __future__ import annotations

import json

from pydantic import BaseModel

from synth_ai.sdk.swarms.types.files import File, FileManifest, FileUploadReceipt, FileUploadStatus
from synth_ai.sdk.swarms.types.launch import (
    FileRef,
    Inputs,
    LaunchRequest,
    Limits,
    Models,
    Project,
    Providers,
    ResolvedModels,
    ResolvedProviders,
)
from synth_ai.sdk.swarms.types.outputs import Output, OutputRef, TextContent, TransferTiming
from synth_ai.sdk.swarms.types.programs import ProgramRunView
from synth_ai.sdk.swarms.types.projects import ProjectCreateRequest
from synth_ai.sdk.swarms.types.runs import (
    LaunchPreflight,
    ResolvedConfiguration,
    RunSnapshot,
    SwarmEvent,
    SwarmResult,
    TaskOutcome,
)
from synth_ai.types.base import EventCursor, Page

from .operations import PREFIX
from .swarms import ALL_OPERATIONS

MODELS: tuple[type[BaseModel], ...] = (
    ProgramRunView,
    FileRef,
    File,
    FileManifest,
    FileUploadReceipt,
    FileUploadStatus,
    Output,
    OutputRef,
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
    ResolvedModels,
    ResolvedProviders,
    LaunchPreflight,
    ResolvedConfiguration,
    RunSnapshot,
    SwarmEvent,
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
                "models": {model.__name__: model.model_json_schema() for model in MODELS},
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
