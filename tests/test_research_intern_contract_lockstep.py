"""Hold the SDK's Intern contracts in lockstep with the backend-authored OpenAPI.

The Intern contracts under ``synth_ai/sdk/research/contracts/`` are hand-written
mirrors of backend pydantic models: nothing generates them, and nothing until now
compared them. That is a silent-drift shape. Because every mirror is
``extra="forbid"``, a field the backend adds does not degrade the SDK -- it makes
``from_wire`` raise on a live response, and only on the code path that populates
the new field. The Async 24/7 cutover hit exactly that with
``host_lease.generation``.

``openapi/research-v1.json`` is the backend-authored bounded contract, exported by
``backend/scripts/export_research_openapi.py`` and already held byte-identical to
the backend artifact by ``testing/scripts/check_research_openapi_contract.py``.
So it is a trustworthy stand-in for the backend models in a repo that cannot
import them, and these tests close the last link in that chain:

    backend contracts -> exported OpenAPI -> vendored contract -> SDK mirrors

Scope is deliberately the Intern surface. The pairings below are the only
hand-maintained thing here -- the SDK prefixes its mirrors ``Intern*`` to keep
them distinct from the SMR nouns, so the names cannot be matched automatically.
Every field set and enum vocabulary on both sides is derived.
"""

from __future__ import annotations

import json
from enum import EnumMeta
from pathlib import Path
from typing import Any, Mapping

import pytest
from pydantic import BaseModel
from synth_ai.sdk.research.contracts import research_intern

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "openapi" / "research-v1.json"

#: SDK mirror -> backend schema in the bounded Research contract.
MIRRORED_MODELS: Mapping[str, str] = {
    "InternAsyncRuntime": "AsyncRuntimeResponse",
    "InternAsyncRuntimeHostLease": "AsyncRuntimeHostLease",
    "InternAsyncRuntimeSpend": "AsyncRuntimeSpend",
    "InternAsyncRuntimeBudget": "AsyncRuntimeBudget",
    "InternAsyncEnsureRequest": "AsyncRuntimeEnsureRequest",
    "InternAsyncCheckpoint": "AsyncCheckpointResponse",
    "InternAsyncBlocker": "AsyncBlockerResponse",
    "InternAsyncJudgmentItem": "AsyncJudgmentItemResponse",
    "InternAsyncEffortWorkSummary": "AsyncEffortWorkSummary",
    "InternAsyncCommandRequest": "InternRuntimeCommandRequest",
    "InternResumeCondition": "ResumeConditionResponse",
    "InternProducedResourceReference": "ProducedResourceReferenceV1",
    "InternSyncSession": "SyncSessionResponse",
    "InternRuntimeBinding": "RuntimeBinding",
}

#: SDK mirror -> backend enum in the bounded Research contract.
MIRRORED_ENUMS: Mapping[str, str] = {
    "InternAsyncStatus": "AsyncStatus",
    "InternSyncStatus": "SyncStatus",
    "InternProducedResourceKind": "ProducedResourceKind",
    "InternAsyncExternalExecutionStatus": "ExternalExecutionStatus",
    "InternAsyncEvidenceReadiness": "EvidenceReadiness",
    "InternAsyncRuntimePhase": "RuntimePhaseWire",
    "InternAsyncStopReason": "StopReason",
    "InternAsyncResumeKind": "ResumeKind",
    "InternAsyncWaitProducer": "WaitProducer",
}


def _schemas() -> Mapping[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return payload["components"]["schemas"]


@pytest.fixture(scope="module")
def schemas() -> Mapping[str, Any]:
    return _schemas()


@pytest.mark.parametrize(("mirror_name", "schema_name"), sorted(MIRRORED_MODELS.items()))
def test_mirrored_model_fields_match_backend_contract(
    schemas: Mapping[str, Any], mirror_name: str, schema_name: str
) -> None:
    """Every mirrored model carries exactly the backend's field names."""

    mirror: type[BaseModel] = getattr(research_intern, mirror_name)
    schema = schemas[schema_name]
    mirror_fields = set(mirror.model_fields)
    backend_fields = set(schema["properties"])
    assert mirror_fields == backend_fields, (
        f"{mirror_name} drifted from backend {schema_name}: "
        f"backend-only={sorted(backend_fields - mirror_fields)} "
        f"(these make from_wire raise on a live response) "
        f"sdk-only={sorted(mirror_fields - backend_fields)}"
    )


@pytest.mark.parametrize(("mirror_name", "schema_name"), sorted(MIRRORED_MODELS.items()))
def test_mirrored_model_required_fields_match_backend_contract(
    schemas: Mapping[str, Any], mirror_name: str, schema_name: str
) -> None:
    """A field the backend always sends is required on the mirror too.

    Optional-where-the-backend-is-required is the quieter half of the same drift:
    it does not raise, it lets a caller read a default for something that is
    never absent and branch on an outcome the backend cannot produce.

    Only that direction is asserted. The mirrors are deliberately *stricter* than
    the contract on the constant discriminators (``schema_version``,
    ``cardinality``, ``instance_kind``, ``leave_safe``), which the backend
    declares with a default and therefore omits from ``required``. Demanding them
    on the wire rejects a truncated response instead of defaulting past it.
    """

    mirror: type[BaseModel] = getattr(research_intern, mirror_name)
    schema = schemas[schema_name]
    mirror_required = {name for name, field in mirror.model_fields.items() if field.is_required()}
    backend_required = set(schema.get("required", ()))
    assert not backend_required - mirror_required, (
        f"{mirror_name} is looser than backend {schema_name}: "
        f"optional here but always sent by the backend="
        f"{sorted(backend_required - mirror_required)}"
    )


@pytest.mark.parametrize(("mirror_name", "schema_name"), sorted(MIRRORED_ENUMS.items()))
def test_mirrored_enum_vocabulary_matches_backend_contract(
    schemas: Mapping[str, Any], mirror_name: str, schema_name: str
) -> None:
    """Every mirrored vocabulary admits exactly the backend's values."""

    mirror: EnumMeta = getattr(research_intern, mirror_name)
    schema = schemas[schema_name]
    mirror_values = {member.value for member in mirror}
    backend_values = set(schema["enum"])
    assert mirror_values == backend_values, (
        f"{mirror_name} drifted from backend {schema_name}: "
        f"backend-only={sorted(backend_values - mirror_values)} "
        f"(these make from_wire raise on a live response) "
        f"sdk-only={sorted(mirror_values - backend_values)}"
    )


def test_every_mirrored_name_resolves() -> None:
    """The pairing table itself cannot rot silently.

    A renamed mirror or a schema dropped from the bounded contract would
    otherwise skip its own coverage rather than fail.
    """

    schemas = _schemas()
    missing_mirrors = sorted(
        name for name in (*MIRRORED_MODELS, *MIRRORED_ENUMS) if not hasattr(research_intern, name)
    )
    missing_schemas = sorted(
        name
        for name in (*MIRRORED_MODELS.values(), *MIRRORED_ENUMS.values())
        if name not in schemas
    )
    assert not missing_mirrors and not missing_schemas, (
        f"lockstep pairing table is stale: missing SDK mirrors={missing_mirrors} "
        f"missing backend schemas={missing_schemas}"
    )
