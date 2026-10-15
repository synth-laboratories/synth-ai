"""Deterministic provisional contract export. See backend SDK index_contract.md.

Backend adoption must own regeneration/parity before native acceptance. This is
not OpenAPI and does not claim implemented backend routes.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from synth_ai.core.http.response import ErrorResponse
from synth_ai.options import MutationOptions, ObservationOptions, ReadOptions, TransferOptions
from synth_ai.sdk.index.types.account import AuthContext, Collection, IndexAccount
from synth_ai.sdk.index.types.annotations import (
    AuthorAnnotations,
    ContributionAnnotations,
    ContributionMetadata,
    ContributionReview,
)
from synth_ai.sdk.index.types.contents import (
    ContentOptions,
    ContentRef,
    ContentRequest,
    ContentResponse,
)
from synth_ai.sdk.index.types.contributions import (
    AttestationReceipt,
    Contribution,
    ContributionDraft,
    ContributionFilter,
    ContributionRevision,
    CorrectionReceipt,
    ResearchCorrection,
    RightsAttestation,
    SubmissionReceipt,
    SubmissionRequest,
)
from synth_ai.sdk.index.types.history import (
    SearchHistoryDeletion,
    SearchHistoryDeletionReceipt,
    SearchHistoryEntry,
    SearchHistoryFilter,
    SearchHistoryPage,
    SearchHistoryRequest,
)
from synth_ai.sdk.index.types.retention import SearchRetention
from synth_ai.sdk.index.types.search import (
    KeyedSearchOptions,
    KeyedSearchRequest,
    PublicSearchOptions,
    PublicSearchRequest,
    SearchEvent,
    SearchResponse,
    SearchSnapshot,
)
from synth_ai.types.account import AccountEntitlements
from synth_ai.types.base import EventCursor, Page
from synth_ai.types.capabilities import Capabilities
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import ControlReceipt, MutationResolution
from synth_ai.types.uploads import (
    ContributionUploadManifest,
    TransferReceipt,
    UploadReceipt,
    UploadStatus,
)
from synth_ai.types.usage import UsageOperation, UsageSummary

from .operations import ALL_OPERATIONS, CONTRACT, PREFIX

MODELS: tuple[type[BaseModel], ...] = (
    AccountEntitlements,
    Capabilities,
    AuthContext,
    AuthorAnnotations,
    ContributionAnnotations,
    ContributionMetadata,
    ContributionReview,
    SearchRetention,
    SearchHistoryDeletion,
    SearchHistoryDeletionReceipt,
    SearchHistoryEntry,
    SearchHistoryFilter,
    SearchHistoryPage,
    SearchHistoryRequest,
    Collection,
    IndexAccount,
    Contribution,
    ContributionDraft,
    ContributionFilter,
    ContributionRevision,
    RightsAttestation,
    AttestationReceipt,
    SubmissionRequest,
    SubmissionReceipt,
    ResearchCorrection,
    CorrectionReceipt,
    ContentRef,
    ContentOptions,
    ContentRequest,
    ContentResponse,
    PublicSearchOptions,
    KeyedSearchOptions,
    PublicSearchRequest,
    KeyedSearchRequest,
    SearchEvent,
    SearchResponse,
    SearchSnapshot,
    MutationIdentity,
    MutationResolution,
    ControlReceipt,
    ContributionUploadManifest,
    UploadStatus,
    UploadReceipt,
    TransferReceipt,
    UsageSummary,
    UsageOperation,
    ErrorResponse,
    ReadOptions,
    MutationOptions,
    ObservationOptions,
    TransferOptions,
    Page[SearchSnapshot],
    Page[SearchEvent, EventCursor],
    Page[Contribution],
    Page[Collection],
    Page[UsageOperation],
)


def export_contract() -> str:
    """Return stable schema/operation facts; no timestamps or reflection."""
    operations = [
        {
            "id": operation.id,
            "method": operation.method,
            "path": PREFIX + operation.path,
            "keyed": operation.keyed,
            "mutation": operation.mutation,
            "required_scope": operation.required_scope,
            "may_charge": operation.may_charge,
            "idempotent": operation.idempotent,
        }
        for operation in ALL_OPERATIONS
    ]
    return (
        json.dumps(
            {
                "schema_version": CONTRACT,
                "status": "sdk_first_pending_backend",
                "operations": operations,
                "models": {model.__name__: model.model_json_schema() for model in MODELS},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    print(export_contract(), end="")
