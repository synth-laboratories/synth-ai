"""Python-only Synth SDK surface."""

from __future__ import annotations

import importlib
from importlib import metadata as _metadata

# Aliased to underscore names so they do not land in the package namespace:
# `synth_ai.Path` / `synth_ai.Any` are import artifacts, not public API, and
# anything reachable from `import synth_ai` but absent from __all__ reads as
# surface a customer may rely on.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from pathlib import Path as _Path
from typing import Any as _Any

try:
    from synth_ai.core.utils.log_filter import (
        install_log_filter as _install_log_filter,
    )

    _install_log_filter()
except Exception:
    pass

try:
    __version__ = _metadata.version("synth-ai")
except _PackageNotFoundError:
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as _toml  # type: ignore[no-redef]  # ty: ignore[unresolved-import]

    try:
        pyproject_path = _Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as fh:
            _pyproject = _toml.load(fh)
        __version__ = str(_pyproject["project"]["version"])
    except Exception:
        __version__ = "0.0.0.dev0"


__all__ = [
    "AsyncResearchInternReactiveSession",
    "AsyncSynthClient",
    "DataBindingCreateRequest",
    "DatasetRevisionCreateRequest",
    "DatasetRevisionFinalizeRequest",
    "DatasetRevisionLifecycleRequest",
    "DatasetRevisionPrepareRequest",
    "DraftDeliveryAuthorityResponse",
    "EnsureDraftDeliveryRequest",
    "FactoryLunaRole",
    "FactoryRoleReceiptMintRequest",
    "FactoryRoleReceiptProvenance",
    "FactoryRoleReceiptResponse",
    "FactoryRoleReceiptRuntimeEvidence",
    "FactoryStorageAuthorityResponse",
    "ResearchApiError",
    "MagiDecisionKind",
    "MagiDecisionReceiptResponse",
    "MagiDecisionRequest",
    "MagiMode",
    "ProjectComputerCleanupReceiptResponse",
    "ProjectComputerCleanupRequest",
    "ProjectComputerExecuteRequest",
    "ProjectComputerInspectRequest",
    "ProjectComputerLeaseAcquireRequest",
    "ProjectComputerLeaseReleaseRequest",
    "ProjectComputerLeaseRenewRequest",
    "ProjectComputerOperationReconcileRequest",
    "ProjectComputerProvisionRequest",
    "ProjectComputerReplaceRequest",
    "WorkspacePushConfirmationReceipt",
    "ResearchClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchLimitExtensionGuardedResumeBlockedError",
    "ResearchLimitExtensionIdempotencyConflictError",
    "ResearchLimitRevisionConflictError",
    "ResearchProjectCreateRequest",
    "ResearchInternProvisionRequest",
    "ResearchInternReactiveSession",
    "ResearchInternEventStreamCursor",
    "ResearchInternEventStreamEvent",
    "ResearchInternEventStreamHeartbeat",
    "ResearchInternResponse",
    "ResearchInternSessionCreateRequest",
    "ResearchInternSessionResponse",
    "ResearchInternStatus",
    "ResearchInternTracePublicationRequest",
    "ResearchInternTracePublicationResponse",
    "ResearchInternTurnControl",
    "ResearchInternTurnRequest",
    "ResearchInternTurnResponse",
    "ResearchSwarmLaunchRequest",
    "ResearchSwarmState",
    "ResearchVisual",
    "ResearchVisualPage",
    "ResearchUnsafeLimitExtensionError",
    "ResearchVisualPatchRequest",
    "ResearchVisualPromotionRequest",
    "ResearchVisualVersions",
    "SynthClient",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AsyncResearchInternReactiveSession": (
        "synth_ai.sdk.research.research_intern",
        "AsyncResearchInternReactiveSession",
    ),
    "DataBindingCreateRequest": (
        "synth_ai.sdk.research.contracts",
        "DataBindingCreateRequest",
    ),
    "DatasetRevisionCreateRequest": (
        "synth_ai.sdk.research.contracts",
        "DatasetRevisionCreateRequest",
    ),
    "DatasetRevisionFinalizeRequest": (
        "synth_ai.sdk.research.contracts",
        "DatasetRevisionFinalizeRequest",
    ),
    "DatasetRevisionLifecycleRequest": (
        "synth_ai.sdk.research.contracts",
        "DatasetRevisionLifecycleRequest",
    ),
    "DatasetRevisionPrepareRequest": (
        "synth_ai.sdk.research.contracts",
        "DatasetRevisionPrepareRequest",
    ),
    "DraftDeliveryAuthorityResponse": (
        "synth_ai.sdk.research.contracts",
        "DraftDeliveryAuthorityResponse",
    ),
    "EnsureDraftDeliveryRequest": (
        "synth_ai.sdk.research.contracts",
        "EnsureDraftDeliveryRequest",
    ),
    "FactoryLunaRole": (
        "synth_ai.sdk.research.contracts",
        "FactoryLunaRole",
    ),
    "FactoryRoleReceiptMintRequest": (
        "synth_ai.sdk.research.contracts",
        "FactoryRoleReceiptMintRequest",
    ),
    "FactoryRoleReceiptProvenance": (
        "synth_ai.sdk.research.contracts",
        "FactoryRoleReceiptProvenance",
    ),
    "FactoryRoleReceiptResponse": (
        "synth_ai.sdk.research.contracts",
        "FactoryRoleReceiptResponse",
    ),
    "FactoryRoleReceiptRuntimeEvidence": (
        "synth_ai.sdk.research.contracts",
        "FactoryRoleReceiptRuntimeEvidence",
    ),
    "FactoryStorageAuthorityResponse": (
        "synth_ai.sdk.research.contracts",
        "FactoryStorageAuthorityResponse",
    ),
    "MagiDecisionRequest": (
        "synth_ai.sdk.research.contracts",
        "MagiDecisionRequest",
    ),
    "MagiDecisionKind": (
        "synth_ai.sdk.research.contracts",
        "MagiDecisionKind",
    ),
    "MagiDecisionReceiptResponse": (
        "synth_ai.sdk.research.contracts",
        "MagiDecisionReceiptResponse",
    ),
    "MagiMode": ("synth_ai.sdk.research.contracts", "MagiMode"),
    "ProjectComputerCleanupReceiptResponse": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerCleanupReceiptResponse",
    ),
    "ProjectComputerCleanupRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerCleanupRequest",
    ),
    "ProjectComputerExecuteRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerExecuteRequest",
    ),
    "ProjectComputerInspectRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerInspectRequest",
    ),
    "ProjectComputerLeaseAcquireRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerLeaseAcquireRequest",
    ),
    "ProjectComputerLeaseReleaseRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerLeaseReleaseRequest",
    ),
    "ProjectComputerLeaseRenewRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerLeaseRenewRequest",
    ),
    "ProjectComputerOperationReconcileRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerOperationReconcileRequest",
    ),
    "ProjectComputerProvisionRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerProvisionRequest",
    ),
    "ProjectComputerReplaceRequest": (
        "synth_ai.sdk.research.contracts",
        "ProjectComputerReplaceRequest",
    ),
    "WorkspacePushConfirmationReceipt": (
        "synth_ai.sdk.research.contracts",
        "WorkspacePushConfirmationReceipt",
    ),
    "ResearchApiError": ("synth_ai.sdk.research.errors", "ResearchApiError"),
    "ResearchConcurrentRunLimitExceededError": (
        "synth_ai.sdk.research.errors",
        "ResearchConcurrentRunLimitExceededError",
    ),
    "ResearchInsufficientCreditsError": (
        "synth_ai.sdk.research.errors",
        "ResearchInsufficientCreditsError",
    ),
    "ResearchLimitExceededError": ("synth_ai.sdk.research.errors", "ResearchLimitExceededError"),
    "ResearchLimitExtensionGuardedResumeBlockedError": (
        "synth_ai.sdk.research.errors",
        "ResearchLimitExtensionGuardedResumeBlockedError",
    ),
    "ResearchLimitExtensionIdempotencyConflictError": (
        "synth_ai.sdk.research.errors",
        "ResearchLimitExtensionIdempotencyConflictError",
    ),
    "ResearchLimitRevisionConflictError": (
        "synth_ai.sdk.research.errors",
        "ResearchLimitRevisionConflictError",
    ),
    "ResearchUnsafeLimitExtensionError": (
        "synth_ai.sdk.research.errors",
        "ResearchUnsafeLimitExtensionError",
    ),
    "SmrApiError": ("synth_ai.sdk.research.errors", "ResearchApiError"),
    "SmrConcurrentRunLimitExceededError": (
        "synth_ai.sdk.research.errors",
        "ResearchConcurrentRunLimitExceededError",
    ),
    "SmrInsufficientCreditsError": (
        "synth_ai.sdk.research.errors",
        "ResearchInsufficientCreditsError",
    ),
    "SmrLimitExceededError": ("synth_ai.sdk.research.errors", "ResearchLimitExceededError"),
    "ResearchClient": ("synth_ai.sdk.research.facade", "ResearchClient"),
    "ResearchProjectCreateRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchProjectCreateRequest",
    ),
    "ResearchInternProvisionRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternProvisionRequest",
    ),
    "ResearchInternReactiveSession": (
        "synth_ai.sdk.research.research_intern",
        "ResearchInternReactiveSession",
    ),
    "ResearchInternEventStreamCursor": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternEventStreamCursor",
    ),
    "ResearchInternEventStreamEvent": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternEventStreamEvent",
    ),
    "ResearchInternEventStreamHeartbeat": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternEventStreamHeartbeat",
    ),
    "ResearchInternResponse": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternResponse",
    ),
    "ResearchInternSessionCreateRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternSessionCreateRequest",
    ),
    "ResearchInternSessionResponse": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternSessionResponse",
    ),
    "ResearchInternStatus": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternStatus",
    ),
    "ResearchInternTracePublicationRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternTracePublicationRequest",
    ),
    "ResearchInternTracePublicationResponse": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternTracePublicationResponse",
    ),
    "ResearchInternTurnControl": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternTurnControl",
    ),
    "ResearchInternTurnRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternTurnRequest",
    ),
    "ResearchInternTurnResponse": (
        "synth_ai.sdk.research.contracts",
        "ResearchInternTurnResponse",
    ),
    "ResearchSwarmLaunchRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchSwarmLaunchRequest",
    ),
    "ResearchSwarmState": (
        "synth_ai.sdk.research.contracts",
        "ResearchSwarmState",
    ),
    "ResearchVisual": ("synth_ai.sdk.research.contracts", "ResearchVisual"),
    "ResearchVisualPage": ("synth_ai.sdk.research.contracts", "ResearchVisualPage"),
    "ResearchVisualPatchRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchVisualPatchRequest",
    ),
    "ResearchVisualPromotionRequest": (
        "synth_ai.sdk.research.contracts",
        "ResearchVisualPromotionRequest",
    ),
    "ResearchVisualVersions": (
        "synth_ai.sdk.research.contracts",
        "ResearchVisualVersions",
    ),
    "SynthClient": ("synth_ai.client", "SynthClient"),
    "AsyncSynthClient": ("synth_ai.client", "AsyncSynthClient"),
}


def __getattr__(name: str) -> _Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
