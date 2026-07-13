"""Hosted optimizer SDK helpers.

Access via ``SynthClient().optimizers``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from pydantic import BaseModel, Field

from synth_ai.sdk.base import SynthBaseClient

__all__ = [
    "AsyncOptimizersClient",
    "OnlineReflexionEvidencePacket",
    "OnlineReflexionEvidenceNotesReview",
    "OnlineReflexionReceiptAudit",
    "OnlineReflexionReceiptAuditSet",
    "OnlineReflexionReceiptBundle",
    "OnlineReflexionReceiptList",
    "OnlineReflexionStartupPreflight",
    "OptimizerBillingFeatureConfig",
    "OptimizerCatalogEntry",
    "OptimizerRun",
    "OptimizerStartupCatalog",
    "OptimizerStartupPreflightError",
    "OptimizersClient",
]

_ONLINE_REFLEXION_RELEASE_LANES: tuple[dict[str, str], ...] = (
    {
        "key": "craftax_rotated_121_125",
        "label": "Craftax rotated 121-125 heldout repeats 2+3",
    },
    {
        "key": "alfworld_6x6_x3",
        "label": "ALFWorld 6/6 matched compare repeated three times",
    },
    {
        "key": "ebr_first_scale_compare",
        "label": "EBR first scale compare",
    },
    {
        "key": "harvey_lab_pilot",
        "label": "Harvey LAB pilot",
    },
    {
        "key": "hosted_staging_smoke",
        "label": "Hosted staging smoke with terminal receipt chain",
    },
)
_ONLINE_REFLEXION_COMPLETE_STATUSES = frozenset(
    {"pass", "passed", "complete", "completed", "ready", "succeeded"}
)


class OptimizerRun(BaseModel):
    """Hosted optimizer run summary returned by the owner API."""

    run_id: str
    algorithm: str
    status: str
    project_id: str | None = None
    finalize_state: str | None = None
    storage_mode: str | None = None
    cursor_seq: int | None = None
    cancellation_requested: bool | None = None
    submitted_at: str | None = None
    terminal_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error: str | None = None
    config: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    backend_projection: dict[str, Any] | None = None
    artifact_handles: dict[str, Any] | None = None


class OptimizerCatalogEntry(BaseModel):
    """One hosted optimizer algorithm advertised by the startup catalog."""

    algorithm: str
    candidate_kinds: list[str] = Field(default_factory=list)
    status: str
    submit_supported: bool


class OptimizerBillingFeatureConfig(BaseModel):
    """Billing feature id metadata for one hosted optimizer algorithm."""

    feature_id: str
    env_override: bool = False


class OptimizerStartupCatalog(BaseModel):
    """Hosted optimizer startup catalog and submit capability preflight."""

    available_algorithms: list[OptimizerCatalogEntry] = Field(default_factory=list)
    org_id: str | None = None
    optimizers_beta_configured: bool = False
    billing_feature_ids: dict[str, OptimizerBillingFeatureConfig] = Field(default_factory=dict)
    billing_feature_ids_configured: dict[str, bool] = Field(default_factory=dict)
    submit_supported: list[str] = Field(default_factory=list)
    online_reflexion_release_evidence: dict[str, Any] = Field(default_factory=dict)


class OnlineReflexionStartupPreflight(BaseModel):
    """Online Reflexion startup readiness check for release automation."""

    ok: bool
    missing_requirements: list[str] = Field(default_factory=list)
    catalog: OptimizerStartupCatalog


class OptimizerStartupPreflightError(RuntimeError):
    """Raised when a hosted optimizer startup preflight requirement is missing."""

    def __init__(self, preflight: OnlineReflexionStartupPreflight) -> None:
        self.preflight = preflight
        requirements = ", ".join(preflight.missing_requirements) or "unknown"
        super().__init__(f"optimizer startup preflight failed: {requirements}")


class OnlineReflexionReceiptBundle(BaseModel):
    """Published online Reflexion receipt bundle for one optimizer run."""

    schema_version: str
    run: dict[str, Any]
    receipt: dict[str, Any]
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    exposures: list[dict[str, Any]] = Field(default_factory=list)
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)


class OnlineReflexionReceiptList(BaseModel):
    """List of published online Reflexion receipt summaries."""

    schema_version: str
    org_id: str
    layer_id: str | None = None
    project_id: str | None = None
    limit: int
    receipts: list[dict[str, Any]] = Field(default_factory=list)


class OnlineReflexionReceiptAudit(BaseModel):
    """Receipt-completeness audit for one online Reflexion optimizer run."""

    schema_version: str
    status: str
    run: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, Any] = Field(default_factory=dict)
    gate_skipped_reason_distribution: dict[str, int] = Field(default_factory=dict)
    missing_commit_event_exposure_ids: list[str] = Field(default_factory=list)
    missing_lever_effect_exposure_ids: list[str] = Field(default_factory=list)
    commit_event_without_exposure_ids: list[str] = Field(default_factory=list)
    lever_effect_without_exposure_ids: list[str] = Field(default_factory=list)
    duplicate_commit_event_exposure_ids: list[str] = Field(default_factory=list)
    summary_count_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    prod_smr_edit_rows: list[dict[str, Any]] = Field(default_factory=list)
    injection_violation_rows: list[dict[str, Any]] = Field(default_factory=list)
    artifact_audit_error: Any | None = None


class OnlineReflexionReceiptAuditSet(BaseModel):
    """Aggregate receipt-completeness audit over a publish-candidate run set."""

    schema_version: str
    status: str
    org_id: str
    selection: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, Any] = Field(default_factory=dict)
    missing_run_ids: list[str] = Field(default_factory=list)
    attention_required_run_ids: list[str] = Field(default_factory=list)
    gate_skipped_reason_distribution: dict[str, int] = Field(default_factory=dict)
    reports: list[OnlineReflexionReceiptAudit] = Field(default_factory=list)


class OnlineReflexionEvidencePacket(BaseModel):
    """Release-readiness packet for online Reflexion evidence and claim review."""

    schema_version: str
    status: str
    public_copy_allowed: bool
    blog_decision_owner: str
    built_at: str
    selection: dict[str, Any] = Field(default_factory=dict)
    claim_gate: dict[str, Any] = Field(default_factory=dict)
    release_gate: dict[str, Any] = Field(default_factory=dict)
    required_evidence: list[dict[str, Any]] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)
    audit: OnlineReflexionReceiptAuditSet
    receipt_summaries: list[dict[str, Any]] = Field(default_factory=list)


class OnlineReflexionEvidenceNotesReview(BaseModel):
    """Local review of online Reflexion evidence notes before receipt assembly."""

    schema_version: str
    status: str
    evidence_lanes_complete: bool
    release_gate_complete: bool
    release_gate: dict[str, Any] = Field(default_factory=dict)
    required_evidence: list[dict[str, Any]] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)


class OptimizersClient(SynthBaseClient):
    """Submit/read hosted optimizer runs, receipts, and online Reflexion audits."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            backend_base=backend_base,
            timeout_seconds=timeout_seconds,
        )
        self._prefix = "/api/v1/optimizers"

    def startup(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> OptimizerStartupCatalog:
        """Return optimizer algorithm availability and billing preflight metadata."""
        payload = self._request(
            "GET",
            f"{self._prefix}/startup",
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OptimizerStartupCatalog, _optimizer_startup_payload(payload))

    def online_reflexion_startup_preflight(
        self,
        *,
        timeout_seconds: float | None = None,
        require_release_metadata: bool = True,
        raise_on_failure: bool = False,
    ) -> OnlineReflexionStartupPreflight:
        """Check whether the backend is ready for hosted online Reflexion release work.

        The release metadata gate verifies the startup schema, release/blog/growth
        gate, required evidence lanes, standard artifacts, owner approval, and the
        EffortBench Chinese-wall marker before any run is submitted.
        """
        catalog = self.startup(timeout_seconds=timeout_seconds)
        preflight = _online_reflexion_startup_preflight(
            catalog,
            require_release_metadata=require_release_metadata,
        )
        if raise_on_failure and not preflight.ok:
            raise OptimizerStartupPreflightError(preflight)
        return preflight

    def validate_online_reflexion_evidence_notes(
        self,
        evidence_notes: Mapping[str, Any] | None = None,
        *,
        evidence_notes_path: str | PathLike[str] | None = None,
    ) -> OnlineReflexionEvidenceNotesReview:
        """Validate Online Reflexion release evidence notes without a backend call.

        Pass either an in-memory ``evidence_notes`` mapping or an
        ``evidence_notes_path`` JSON file. The two inputs are mutually exclusive.
        """
        return _validate_online_reflexion_evidence_notes(
            _load_online_reflexion_evidence_notes(
                evidence_notes,
                evidence_notes_path=evidence_notes_path,
            )
        )

    def get_run(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> OptimizerRun:
        """Retrieve one hosted optimizer run."""
        payload = self._request(
            "GET",
            f"{self._prefix}/runs/{run_id}",
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OptimizerRun, payload)

    def submit_run(
        self,
        *,
        algorithm: str = "gepa",
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        config_toml: str | None = None,
        config_json: dict[str, Any] | None = None,
        container_pool: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> OptimizerRun:
        """Submit a hosted optimizer run through the optimizer owner API."""
        body: dict[str, Any] = {"algorithm": algorithm}
        if run_id:
            body["run_id"] = run_id
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        if project_id:
            body["project_id"] = project_id
        if config_toml is not None:
            body["config_toml"] = config_toml
        if config_json is not None:
            body["config_json"] = config_json
        if container_pool is not None:
            body["container_pool"] = container_pool
        payload = self._request(
            "POST",
            f"{self._prefix}/runs",
            json_body=body,
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OptimizerRun, payload)

    def submit_online_reflexion(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> OptimizerRun:
        """Submit a hosted online Reflexion optimizer run."""
        return self.submit_run(
            algorithm="online-reflexion",
            run_id=run_id,
            idempotency_key=idempotency_key,
            project_id=project_id,
            config_json=config,
            container_pool=container_pool,
            timeout_seconds=timeout_seconds,
        )

    def online_reflexion_receipt(
        self,
        run_id: str,
        *,
        exposure_limit: int = 500,
        outcome_limit: int = 500,
        timeout_seconds: float | None = None,
    ) -> OnlineReflexionReceiptBundle:
        """Return the published online Reflexion receipt bundle for one run."""
        payload = self._request(
            "GET",
            f"{self._prefix}/runs/{run_id}/online-reflexion/receipt",
            params={
                "exposure_limit": exposure_limit,
                "outcome_limit": outcome_limit,
            },
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OnlineReflexionReceiptBundle, payload)

    def online_reflexion_receipt_audit(
        self,
        run_id: str,
        *,
        strict: bool = False,
        timeout_seconds: float | None = None,
    ) -> OnlineReflexionReceiptAudit:
        """Audit receipt completeness for one online Reflexion run."""
        payload = self._request(
            "GET",
            f"{self._prefix}/runs/{run_id}/online-reflexion/receipt-audit",
            params={"strict": strict},
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OnlineReflexionReceiptAudit, payload)

    def online_reflexion_receipt_audits(
        self,
        *,
        run_ids: list[str] | None = None,
        layer_id: str | None = None,
        project_id: str | None = None,
        strict: bool = False,
        limit: int = 50,
        timeout_seconds: float | None = None,
    ) -> OnlineReflexionReceiptAuditSet:
        """Audit receipt completeness for a publish-candidate run set."""
        params: dict[str, Any] = {
            "strict": strict,
            "limit": limit,
        }
        if run_ids:
            params["run_ids"] = ",".join(run_ids)
        if layer_id:
            params["layer_id"] = layer_id
        if project_id:
            params["project_id"] = project_id
        payload = self._request(
            "GET",
            f"{self._prefix}/online-reflexion/receipt-audits",
            params=params,
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OnlineReflexionReceiptAuditSet, payload)

    def online_reflexion_receipts(
        self,
        *,
        layer_id: str | None = None,
        project_id: str | None = None,
        include_summary: bool = False,
        limit: int = 50,
        timeout_seconds: float | None = None,
    ) -> OnlineReflexionReceiptList:
        """List published online Reflexion receipt summaries."""
        params: dict[str, Any] = {
            "include_summary": include_summary,
            "limit": limit,
        }
        if layer_id:
            params["layer_id"] = layer_id
        if project_id:
            params["project_id"] = project_id
        payload = self._request(
            "GET",
            f"{self._prefix}/online-reflexion/receipts",
            params=params,
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(OnlineReflexionReceiptList, payload)

    def online_reflexion_evidence_packet(
        self,
        *,
        run_ids: list[str] | None = None,
        layer_id: str | None = None,
        project_id: str | None = None,
        evidence_notes: Mapping[str, Any] | None = None,
        evidence_notes_path: str | PathLike[str] | None = None,
        blog_decision_owner: str = "Josh",
        blog_approved_by_owner: bool = False,
        include_receipt_summaries: bool = True,
        limit: int = 50,
        timeout_seconds: float | None = None,
    ) -> OnlineReflexionEvidencePacket:
        """Build a release packet from receipt audits and structured lane proof.

        Pass either an in-memory ``evidence_notes`` mapping or an
        ``evidence_notes_path`` JSON file. Path loading happens before backend
        receipt reads so malformed local release evidence fails early.
        """
        evidence_notes_payload = _load_online_reflexion_evidence_notes(
            evidence_notes,
            evidence_notes_path=evidence_notes_path,
        )
        audit = self.online_reflexion_receipt_audits(
            run_ids=run_ids,
            layer_id=layer_id,
            project_id=project_id,
            strict=False,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )
        receipt_summaries: list[dict[str, Any]] = []
        if include_receipt_summaries and (layer_id or project_id or not run_ids):
            receipts = self.online_reflexion_receipts(
                layer_id=layer_id,
                project_id=project_id,
                include_summary=True,
                limit=limit,
                timeout_seconds=timeout_seconds,
            )
            receipt_summaries = receipts.receipts
        return _build_online_reflexion_evidence_packet(
            audit=audit,
            receipt_summaries=receipt_summaries,
            evidence_notes=evidence_notes_payload,
            blog_decision_owner=blog_decision_owner,
            blog_approved_by_owner=blog_approved_by_owner,
        )


class AsyncOptimizersClient:
    """Async adapter over :class:`OptimizersClient`."""

    def __init__(self, sync_client: OptimizersClient) -> None:
        self._sync_client = sync_client

    async def startup(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> OptimizerStartupCatalog:
        """Return optimizer algorithm availability and billing preflight metadata."""
        return await asyncio.to_thread(
            self._sync_client.startup,
            timeout_seconds=timeout_seconds,
        )

    async def online_reflexion_startup_preflight(
        self,
        *,
        timeout_seconds: float | None = None,
        require_release_metadata: bool = True,
        raise_on_failure: bool = False,
    ) -> OnlineReflexionStartupPreflight:
        """Check whether the backend is ready for hosted online Reflexion release work."""
        return await asyncio.to_thread(
            self._sync_client.online_reflexion_startup_preflight,
            timeout_seconds=timeout_seconds,
            require_release_metadata=require_release_metadata,
            raise_on_failure=raise_on_failure,
        )

    async def validate_online_reflexion_evidence_notes(
        self,
        evidence_notes: Mapping[str, Any] | None = None,
        *,
        evidence_notes_path: str | PathLike[str] | None = None,
    ) -> OnlineReflexionEvidenceNotesReview:
        """Validate Online Reflexion release evidence notes without a backend call."""
        return await asyncio.to_thread(
            self._sync_client.validate_online_reflexion_evidence_notes,
            evidence_notes,
            evidence_notes_path=evidence_notes_path,
        )

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._sync_client, name)
        if callable(attr):

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            return _wrapped
        return attr


def _optimizer_startup_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "available_algorithms": [],
            "optimizers_beta_configured": False,
            "billing_feature_ids": {},
            "billing_feature_ids_configured": {},
            "submit_supported": [],
        }
    raw_algorithms = payload.get("available_algorithms")
    available_algorithms: list[dict[str, Any]] = []
    if isinstance(raw_algorithms, Sequence) and not isinstance(raw_algorithms, str):
        for raw_entry in raw_algorithms:
            if not isinstance(raw_entry, Mapping):
                continue
            algorithm = str(raw_entry.get("algorithm") or "").strip()
            status = str(raw_entry.get("status") or "").strip()
            if not algorithm:
                continue
            raw_candidate_kinds = raw_entry.get("candidate_kinds")
            candidate_kinds: list[str] = []
            if isinstance(raw_candidate_kinds, Sequence) and not isinstance(
                raw_candidate_kinds, str
            ):
                candidate_kinds = [str(item) for item in raw_candidate_kinds]
            available_algorithms.append(
                {
                    "algorithm": algorithm,
                    "candidate_kinds": candidate_kinds,
                    "status": status,
                    "submit_supported": status == "available" and algorithm != "ohco",
                }
            )
    return {
        "available_algorithms": available_algorithms,
        "org_id": _optional_text(payload.get("org_id")),
        "optimizers_beta_configured": payload.get("optimizers_beta_configured") is True,
        "billing_feature_ids": _optimizer_billing_feature_ids(
            payload.get("billing_feature_ids")
        ),
        "billing_feature_ids_configured": _optimizer_billing_feature_ids_configured(
            payload.get("billing_feature_ids_configured")
        ),
        "online_reflexion_release_evidence": _object_payload(
            payload.get("online_reflexion_release_evidence")
        ),
        "submit_supported": [
            entry["algorithm"]
            for entry in available_algorithms
            if entry["submit_supported"] is True
        ],
    }


def _sequence_payload(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _online_reflexion_startup_preflight(
    catalog: OptimizerStartupCatalog,
    *,
    require_release_metadata: bool,
) -> OnlineReflexionStartupPreflight:
    missing: list[str] = []
    online_reflexion_available = any(
        entry.algorithm == "online-reflexion"
        and entry.status == "available"
        and entry.submit_supported is True
        for entry in catalog.available_algorithms
    )
    if not online_reflexion_available:
        missing.append("online-reflexion algorithm is not submit-supported")
    if not require_release_metadata:
        return OnlineReflexionStartupPreflight(
            ok=not missing,
            missing_requirements=missing,
            catalog=catalog,
        )

    release_evidence = catalog.online_reflexion_release_evidence
    if not release_evidence:
        missing.append("online_reflexion_release_evidence metadata is not advertised")
        return OnlineReflexionStartupPreflight(
            ok=False,
            missing_requirements=missing,
            catalog=catalog,
        )

    if release_evidence.get("schema_version") != "online_reflexion_release_evidence.v1":
        missing.append("online_reflexion_release_evidence schema_version is not v1")
    if release_evidence.get("release_gate_key") != "release_blog_growth":
        missing.append("online_reflexion release_gate_key is not release_blog_growth")

    lane_keys = {
        str(item.get("key"))
        for item in _sequence_payload(release_evidence.get("required_lanes"))
        if isinstance(item, Mapping) and item.get("key")
    }
    for key in (
        "craftax_rotated_121_125",
        "alfworld_6x6_x3",
        "ebr_first_scale_compare",
        "harvey_lab_pilot",
        "hosted_staging_smoke",
    ):
        if key not in lane_keys:
            missing.append(f"online_reflexion release lane missing: {key}")

    release_checks = _sequence_payload(
        release_evidence.get("release_gate_required_checks")
    )
    if not release_checks:
        missing.append("online_reflexion release_gate_required_checks is empty")

    standard_artifacts = release_evidence.get("standard_artifacts")
    if not isinstance(standard_artifacts, Mapping):
        missing.append("online_reflexion standard_artifacts is not advertised")
    else:
        for key in ("events", "exposures", "lever_effects", "summary"):
            if key not in standard_artifacts:
                missing.append(f"online_reflexion standard artifact missing: {key}")

    if release_evidence.get("public_copy_requires_owner_approval") is not True:
        missing.append("online_reflexion owner approval requirement is not advertised")
    if release_evidence.get("effortbench_cookbook_chinese_wall") != "grader_only":
        missing.append("online_reflexion EffortBench Chinese-wall marker is not grader_only")

    return OnlineReflexionStartupPreflight(
        ok=not missing,
        missing_requirements=missing,
        catalog=catalog,
    )


def _optimizer_billing_feature_ids(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for algorithm, raw_entry in value.items():
        key = str(algorithm)
        if isinstance(raw_entry, str):
            feature_id = raw_entry.strip()
            if feature_id:
                parsed[key] = {"feature_id": feature_id, "env_override": False}
        elif isinstance(raw_entry, Mapping):
            feature_id = _optional_text(raw_entry.get("feature_id"))
            if feature_id:
                parsed[key] = {
                    "feature_id": feature_id,
                    "env_override": raw_entry.get("env_override") is True,
                }
    return parsed


def _optimizer_billing_feature_ids_configured(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, bool] = {}
    for algorithm, configured in value.items():
        if isinstance(configured, bool):
            parsed[str(algorithm)] = configured
    return parsed


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _object_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_online_reflexion_evidence_notes(
    evidence_notes: Mapping[str, Any] | None = None,
    *,
    evidence_notes_path: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    if evidence_notes is not None and evidence_notes_path is not None:
        raise ValueError("evidence_notes and evidence_notes_path are mutually exclusive")
    if evidence_notes_path is None:
        return dict(evidence_notes or {})
    with open(evidence_notes_path, encoding="utf-8") as evidence_file:
        parsed = json.load(evidence_file)
    if not isinstance(parsed, Mapping):
        raise ValueError("evidence_notes_path must contain a JSON object")
    return dict(parsed)


def _validate_online_reflexion_evidence_notes(
    evidence_notes: Mapping[str, Any],
) -> OnlineReflexionEvidenceNotesReview:
    required_evidence: list[dict[str, Any]] = []
    remaining: list[str] = []
    release_review = _release_gate_review(evidence_notes.get("release_blog_growth"))
    for lane in _ONLINE_REFLEXION_RELEASE_LANES:
        note = evidence_notes.get(lane["key"])
        lane_review = _evidence_lane_review(lane["key"], note)
        lane_state = lane_review["state"]
        required_evidence.append(
            {
                **lane,
                "state": lane_state,
                "validation": lane_review,
                "evidence": note,
            }
        )
        if lane_state != "complete":
            remaining.append(f"attach complete evidence for {lane['label']}")
    if release_review["state"] != "complete":
        remaining.append("complete release/blog/growth readiness evidence")
    return OnlineReflexionEvidenceNotesReview(
        schema_version="online_reflexion_evidence_notes_review.v1",
        status="pass" if not remaining else "attention_required",
        evidence_lanes_complete=all(
            item["state"] == "complete" for item in required_evidence
        ),
        release_gate_complete=release_review["state"] == "complete",
        release_gate=release_review,
        required_evidence=required_evidence,
        remaining=remaining,
    )


def _build_online_reflexion_evidence_packet(
    *,
    audit: OnlineReflexionReceiptAuditSet,
    receipt_summaries: list[dict[str, Any]],
    evidence_notes: dict[str, Any],
    blog_decision_owner: str,
    blog_approved_by_owner: bool,
) -> OnlineReflexionEvidencePacket:
    from datetime import UTC, datetime

    required_evidence = []
    remaining: list[str] = []
    release_review = _release_gate_review(evidence_notes.get("release_blog_growth"))
    for lane in _ONLINE_REFLEXION_RELEASE_LANES:
        note = evidence_notes.get(lane["key"])
        lane_review = _evidence_lane_review(lane["key"], note)
        lane_state = lane_review["state"]
        required_evidence.append(
            {
                **lane,
                "state": lane_state,
                "validation": lane_review,
                "evidence": note,
            }
        )
        if lane_state != "complete":
            remaining.append(f"attach complete evidence for {lane['label']}")
    if release_review["state"] != "complete":
        remaining.append("complete release/blog/growth readiness evidence")

    publish_candidate_count = len(audit.reports)
    receipt_audit_passed = audit.status == "pass"
    no_missing_runs = len(audit.missing_run_ids) == 0
    no_attention_required_runs = len(audit.attention_required_run_ids) == 0
    has_publish_candidates = publish_candidate_count > 0
    evidence_lanes_complete = all(item["state"] == "complete" for item in required_evidence)
    release_gate_complete = release_review["state"] == "complete"
    if not has_publish_candidates:
        remaining.append("select at least one hosted online Reflexion publish-candidate run")
    if not receipt_audit_passed:
        remaining.append("clear online Reflexion receipt-completeness audit")
    if not no_missing_runs:
        remaining.append("resolve missing online Reflexion run receipts")
    if not no_attention_required_runs:
        remaining.append("resolve attention-required receipt audits")
    if not blog_approved_by_owner:
        remaining.append(f"obtain {blog_decision_owner} blog/release approval before public copy")

    technical_ready = all(
        [
            receipt_audit_passed,
            no_missing_runs,
            no_attention_required_runs,
            has_publish_candidates,
            evidence_lanes_complete,
            release_gate_complete,
        ]
    )
    status = "ready" if technical_ready and blog_approved_by_owner else "ready_for_owner_review" if technical_ready else "not_ready"
    claim_gate = {
        "receipt_audit_passed": receipt_audit_passed,
        "no_missing_runs": no_missing_runs,
        "no_attention_required_runs": no_attention_required_runs,
        "has_publish_candidates": has_publish_candidates,
        "publish_candidate_count": publish_candidate_count,
        "evidence_lanes_complete": evidence_lanes_complete,
        "release_gate_complete": release_gate_complete,
        "blog_owner_review_passed": blog_approved_by_owner,
    }
    return OnlineReflexionEvidencePacket(
        schema_version="online_reflexion_evidence_packet.v1",
        status=status,
        public_copy_allowed=status == "ready",
        blog_decision_owner=blog_decision_owner,
        built_at=datetime.now(UTC).isoformat(),
        selection=audit.selection,
        claim_gate=claim_gate,
        release_gate=release_review,
        required_evidence=required_evidence,
        remaining=remaining,
        audit=audit,
        receipt_summaries=receipt_summaries,
    )


def _release_gate_review(note: Any) -> dict[str, Any]:
    if note is None:
        return {
            "state": "missing",
            "missing_requirements": ["attach release_blog_growth readiness evidence"],
            "checks": {},
        }
    if not isinstance(note, Mapping):
        return {
            "state": "attached",
            "missing_requirements": [
                "replace release_blog_growth with a structured evidence object"
            ],
            "checks": {"structured_object": False},
        }

    checks: dict[str, Any] = {}
    missing: list[str] = []
    status = str(note.get("status") or "").strip().lower()
    review_complete = note.get("ok") is True or status in _ONLINE_REFLEXION_COMPLETE_STATUSES
    _require(
        review_complete,
        "review_complete",
        "set release_blog_growth ok=true or status=pass/complete/succeeded",
        checks,
        missing,
    )
    _require(
        _truthy(note, "public_docs_ready", "docs_ready"),
        "public_docs_ready",
        "prove public docs/runbook updates are ready",
        checks,
        missing,
    )
    _require(
        _truthy(note, "sdk_cli_stack_ready", "operator_paths_ready"),
        "sdk_cli_stack_ready",
        "prove SDK, CLI, and Stack operator paths are ready",
        checks,
        missing,
    )
    _require(
        _truthy(note, "changelog_ready", "release_notes_ready"),
        "changelog_ready",
        "prove changelog or release notes are ready",
        checks,
        missing,
    )
    _require(
        _truthy(note, "blog_claims_match_evidence", "blog_evidence_matched"),
        "blog_claims_match_evidence",
        "prove blog claims are matched to receipt-backed evidence",
        checks,
        missing,
    )
    _require(
        _truthy(note, "growth_plan_ready", "launch_growth_ready"),
        "growth_plan_ready",
        "prove launch/growth plan is ready",
        checks,
        missing,
    )
    _require(
        _truthy(note, "effortbench_chinese_wall_reviewed", "no_effortbench_cookbook_leak"),
        "effortbench_chinese_wall_reviewed",
        "prove EffortBench cookbook/grader-only materials did not leak into claims",
        checks,
        missing,
    )
    _require(
        _evidence_count(
            note,
            "doc_paths",
            "release_note_paths",
            "blog_paths",
            "growth_paths",
            "artifact_paths",
        )
        >= 1,
        "release_artifact_reference_present",
        "attach at least one release/blog/growth artifact path",
        checks,
        missing,
    )

    return {
        "state": "complete" if not missing else "attached",
        "missing_requirements": missing,
        "checks": checks,
        "evidence": dict(note),
    }


def _evidence_lane_review(lane_key: str, note: Any) -> dict[str, Any]:
    if note is None:
        return {
            "state": "missing",
            "missing_requirements": ["attach structured lane evidence"],
            "checks": {},
        }
    if not isinstance(note, Mapping):
        return {
            "state": "attached",
            "missing_requirements": ["replace loose evidence with a structured evidence object"],
            "checks": {"structured_object": False},
        }

    checks: dict[str, Any] = {}
    missing: list[str] = []
    status = str(note.get("status") or "").strip().lower()
    review_complete = note.get("ok") is True or status in _ONLINE_REFLEXION_COMPLETE_STATUSES
    checks["review_complete"] = review_complete
    if not review_complete:
        missing.append("set ok=true or status=pass/complete/succeeded after lane review")

    if lane_key == "craftax_rotated_121_125":
        _require(
            _heldout_window_is_121_125(note),
            "heldout_window_121_125",
            "prove heldout window is seeds 121-125",
            checks,
            missing,
        )
        _require(
            _evidence_count(note, "run_ids", "artifact_dirs", "receipt_run_ids") >= 2
            or _number_at_least(note, 2, "repeat_count", "repeats", "repeats_passed"),
            "repeats_2_and_3_present",
            "attach Craftax repeat 2 and repeat 3 run/artifact ids",
            checks,
            missing,
        )
        _require(
            _truthy(note, "ci_excludes_zero", "bootstrap_ci_excludes_zero"),
            "ci_excludes_zero",
            "prove heldout bootstrap CI excludes zero",
            checks,
            missing,
        )
        _require(
            _percent_at_most(note, 15.0, "per_inject_harm_pct", "harm_pct", "per_inject_harm"),
            "per_inject_harm_within_bound",
            "prove per-inject harm is <= 15%",
            checks,
            missing,
        )
        _require(
            _truthy(
                note,
                "zero_invalid_injections",
                "zero_ceiling_or_no_failure_injects",
                "zero_injects_at_ceiling",
            ),
            "zero_invalid_injections",
            "prove zero injections on no-failure/at-ceiling trials",
            checks,
            missing,
        )
    elif lane_key == "alfworld_6x6_x3":
        _require(
            _number_at_least(note, 6, "matched_tasks", "task_count", "tasks_matched"),
            "six_of_six_matched",
            "prove ALFWorld used the full 6/6 matched task set",
            checks,
            missing,
        )
        _require(
            _evidence_count(note, "run_ids", "artifact_dirs", "receipt_run_ids") >= 3
            or _number_at_least(note, 3, "clean_repeats", "repeat_count", "repeats"),
            "three_clean_repeats",
            "attach three clean ALFWorld repeat run/artifact ids",
            checks,
            missing,
        )
        _require(
            _truthy(note, "no_truncated_runs", "truncated_runs_discarded", "clean_verdict"),
            "no_truncated_runs_in_verdict",
            "prove truncated ALFWorld runs were discarded from the verdict",
            checks,
            missing,
        )
        _require(
            bool(str(note.get("verdict") or "").strip()),
            "verdict_recorded",
            "record the ALFWorld verdict, even if task_success is flat",
            checks,
            missing,
        )
    elif lane_key == "ebr_first_scale_compare":
        _require(
            _truthy(note, "scale_compare", "scaled_compare", "first_scale_compare"),
            "scale_compare_present",
            "prove EBR ran a scale compare, not only a smoke",
            checks,
            missing,
        )
        _require(
            _evidence_count(note, "run_ids", "artifact_dirs", "receipt_run_ids") >= 1,
            "run_id_present",
            "attach the EBR scale compare run/artifact id",
            checks,
            missing,
        )
        _require(
            bool(str(note.get("verdict") or "").strip()),
            "verdict_recorded",
            "record the EBR scale-compare verdict",
            checks,
            missing,
        )
    elif lane_key == "harvey_lab_pilot":
        _require(
            str(note.get("split") or "").strip().lower() == "tax",
            "tax_split",
            "prove Harvey LAB pilot used the Tax split",
            checks,
            missing,
        )
        _require(
            _number_at_least(note, 25, "train_count", "train", "train_examples"),
            "train_25",
            "prove Harvey LAB pilot used 25 train examples",
            checks,
            missing,
        )
        _require(
            _number_at_least(note, 9, "heldout_count", "heldout", "heldout_examples"),
            "heldout_9",
            "prove Harvey LAB pilot used 9 heldout examples",
            checks,
            missing,
        )
        _require(
            _truthy(note, "criterion_signals_mapped", "judge_criteria_mapped"),
            "criteria_to_failure_signals",
            "prove LAB judge criteria were mapped to typed failure signals",
            checks,
            missing,
        )
        _require(
            _evidence_count(note, "run_ids", "artifact_dirs", "receipt_run_ids") >= 1,
            "run_id_present",
            "attach the Harvey LAB pilot run/artifact id",
            checks,
            missing,
        )
    elif lane_key == "hosted_staging_smoke":
        _require(
            str(note.get("environment") or note.get("env") or "").strip().lower() == "staging",
            "staging_environment",
            "prove the hosted smoke ran against staging",
            checks,
            missing,
        )
        _require(
            str(note.get("terminal_status") or note.get("status") or "").strip().lower()
            in {"succeeded", "success"},
            "terminal_success",
            "prove the hosted run reached terminal success",
            checks,
            missing,
        )
        _require(
            _evidence_count(note, "run_id", "run_ids", "receipt_run_ids") >= 1,
            "run_id_present",
            "attach the hosted staging smoke run id",
            checks,
            missing,
        )
        _require(
            _truthy(note, "receipt_audit_passed", "strict_receipt_audit_passed"),
            "strict_receipt_audit_passed",
            "prove strict receipt audit passed",
            checks,
            missing,
        )
        _require(
            _truthy(note, "standard_artifacts_present", "standard_bundle_present"),
            "standard_artifacts_present",
            "prove the standard artifact bundle is present",
            checks,
            missing,
        )
        _require(
            _truthy(note, "never_blocks_receipt_present", "policy_never_blocks_proven"),
            "never_blocks_proven",
            "prove policy-never-blocks with a receipt",
            checks,
            missing,
        )

    return {
        "state": "complete" if not missing else "attached",
        "missing_requirements": missing,
        "checks": checks,
    }


def _require(
    condition: bool,
    check_name: str,
    missing_requirement: str,
    checks: dict[str, Any],
    missing: list[str],
) -> None:
    checks[check_name] = condition
    if not condition:
        missing.append(missing_requirement)


def _truthy(note: Mapping[str, Any], *keys: str) -> bool:
    return any(note.get(key) is True for key in keys)


def _number_at_least(note: Mapping[str, Any], minimum: float, *keys: str) -> bool:
    for key in keys:
        value = _as_float(note.get(key))
        if value is not None and value >= minimum:
            return True
    return False


def _percent_at_most(note: Mapping[str, Any], maximum_pct: float, *keys: str) -> bool:
    for key in keys:
        value = _as_float(note.get(key))
        if value is None:
            continue
        pct = (
            value
            if key.endswith("_pct") or key == "harm_pct"
            else value * 100.0
            if 0.0 <= value <= 1.0
            else value
        )
        if pct <= maximum_pct:
            return True
    return False


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _evidence_count(note: Mapping[str, Any], *keys: str) -> int:
    count = 0
    for key in keys:
        value = note.get(key)
        if isinstance(value, str) and value.strip():
            count += 1
        elif isinstance(value, Sequence) and not isinstance(value, str):
            count += sum(1 for item in value if str(item or "").strip())
    return count


def _heldout_window_is_121_125(note: Mapping[str, Any]) -> bool:
    window = str(note.get("heldout_window") or note.get("window") or "").replace(" ", "")
    if window in {"121-125", "121..125", "121:125"}:
        return True
    seeds = note.get("seeds") or note.get("heldout_seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, str):
        try:
            return [int(seed) for seed in seeds] == [121, 122, 123, 124, 125]
        except (TypeError, ValueError):
            return False
    return False
