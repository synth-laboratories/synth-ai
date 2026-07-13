"""Typed SDK mirror of the backend SMR operator-evidence contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _mapping_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [_mapping(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)]


def _text(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_text(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _datetime(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    raise ValueError(f"{field_name} must be an ISO-8601 datetime")


def _optional_datetime(value: object, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _datetime(value, field_name=field_name)


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    normalized = value or 0
    if not isinstance(normalized, (int, str)):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list")
    return tuple(str(item) for item in value)


@dataclass(frozen=True, slots=True)
class OperatorEvidenceDiagnostic:
    code: str
    severity: str
    component: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> OperatorEvidenceDiagnostic:
        mapping = _mapping(payload, field_name="operator evidence diagnostic")
        return cls(
            code=_text(mapping.get("code"), field_name="diagnostic.code"),
            severity=_text(mapping.get("severity"), field_name="diagnostic.severity"),
            component=_text(mapping.get("component"), field_name="diagnostic.component"),
            message=_text(mapping.get("message"), field_name="diagnostic.message"),
            detail=_mapping(mapping.get("detail") or {}, field_name="diagnostic.detail"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "component": self.component,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class ProjectionEvidenceReceipt:
    receipt_id: str
    projection_name: str
    subject_id: str
    source: str
    status: str
    observed_at: datetime
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectionEvidenceReceipt:
        mapping = _mapping(payload, field_name="projection evidence receipt")
        return cls(
            receipt_id=_text(mapping.get("receipt_id"), field_name="receipt.receipt_id"),
            projection_name=_text(
                mapping.get("projection_name"), field_name="receipt.projection_name"
            ),
            subject_id=_text(mapping.get("subject_id"), field_name="receipt.subject_id"),
            source=_text(mapping.get("source"), field_name="receipt.source"),
            status=_text(mapping.get("status"), field_name="receipt.status"),
            observed_at=_datetime(mapping.get("observed_at"), field_name="receipt.observed_at"),
            detail=_mapping(mapping.get("detail") or {}, field_name="receipt.detail"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "projection_name": self.projection_name,
            "subject_id": self.subject_id,
            "source": self.source,
            "status": self.status,
            "observed_at": self.observed_at.isoformat(),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class TraceCoverageEvidence:
    trace_count: int
    raw_trace_index_count: int
    durable_trace_count: int
    verified_trace_count: int
    unverified_trace_count: int
    missing_trace_count: int
    artifact_trace_count: int
    task_output_trace_count: int
    traces: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[OperatorEvidenceDiagnostic, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> TraceCoverageEvidence:
        mapping = _mapping(payload, field_name="trace coverage")
        return cls(
            trace_count=_integer(mapping.get("trace_count"), field_name="trace_count"),
            raw_trace_index_count=_integer(
                mapping.get("raw_trace_index_count"), field_name="raw_trace_index_count"
            ),
            durable_trace_count=_integer(
                mapping.get("durable_trace_count"), field_name="durable_trace_count"
            ),
            verified_trace_count=_integer(
                mapping.get("verified_trace_count"), field_name="verified_trace_count"
            ),
            unverified_trace_count=_integer(
                mapping.get("unverified_trace_count"),
                field_name="unverified_trace_count",
            ),
            missing_trace_count=_integer(
                mapping.get("missing_trace_count"), field_name="missing_trace_count"
            ),
            artifact_trace_count=_integer(
                mapping.get("artifact_trace_count"), field_name="artifact_trace_count"
            ),
            task_output_trace_count=_integer(
                mapping.get("task_output_trace_count"),
                field_name="task_output_trace_count",
            ),
            traces=tuple(_mapping_list(mapping.get("traces"), field_name="traces")),
            diagnostics=tuple(
                OperatorEvidenceDiagnostic.from_wire(item)
                for item in list(mapping.get("diagnostics") or [])
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "trace_count": self.trace_count,
            "raw_trace_index_count": self.raw_trace_index_count,
            "durable_trace_count": self.durable_trace_count,
            "verified_trace_count": self.verified_trace_count,
            "unverified_trace_count": self.unverified_trace_count,
            "missing_trace_count": self.missing_trace_count,
            "artifact_trace_count": self.artifact_trace_count,
            "task_output_trace_count": self.task_output_trace_count,
            "traces": [dict(item) for item in self.traces],
            "diagnostics": [item.to_wire() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class TranscriptCoverageEvidence:
    event_count: int
    participant_session_count: int
    counts_by_kind: dict[str, int]
    participant_session_ids: tuple[str, ...]
    thread_ids: tuple[str, ...]
    turn_ids: tuple[str, ...]
    latest_event_at: datetime | None = None
    diagnostics: tuple[OperatorEvidenceDiagnostic, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> TranscriptCoverageEvidence:
        mapping = _mapping(payload, field_name="transcript coverage")
        raw_counts = _mapping(mapping.get("counts_by_kind") or {}, field_name="counts_by_kind")
        return cls(
            event_count=_integer(mapping.get("event_count"), field_name="event_count"),
            participant_session_count=_integer(
                mapping.get("participant_session_count"),
                field_name="participant_session_count",
            ),
            counts_by_kind={key: int(value) for key, value in raw_counts.items()},
            participant_session_ids=_string_tuple(
                mapping.get("participant_session_ids"),
                field_name="participant_session_ids",
            ),
            thread_ids=_string_tuple(mapping.get("thread_ids"), field_name="thread_ids"),
            turn_ids=_string_tuple(mapping.get("turn_ids"), field_name="turn_ids"),
            latest_event_at=_optional_datetime(
                mapping.get("latest_event_at"), field_name="latest_event_at"
            ),
            diagnostics=tuple(
                OperatorEvidenceDiagnostic.from_wire(item)
                for item in list(mapping.get("diagnostics") or [])
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "participant_session_count": self.participant_session_count,
            "counts_by_kind": dict(self.counts_by_kind),
            "participant_session_ids": list(self.participant_session_ids),
            "thread_ids": list(self.thread_ids),
            "turn_ids": list(self.turn_ids),
            "latest_event_at": (
                self.latest_event_at.isoformat() if self.latest_event_at is not None else None
            ),
            "diagnostics": [item.to_wire() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class ReportBenchWitnessEvidence:
    status: str
    witness_role: str
    mutation_policy: str
    source_authority_version: str | None = None
    source_row_ids: dict[str, Any] = field(default_factory=dict)
    terminal_truth: dict[str, Any] = field(default_factory=dict)
    task_lifecycle_summary: dict[str, int] = field(default_factory=dict)
    validation_summary: dict[str, int] = field(default_factory=dict)
    evidence_summary: dict[str, int] = field(default_factory=dict)
    reportbench_evidence: tuple[dict[str, Any], ...] = ()
    verifier_verdicts: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[OperatorEvidenceDiagnostic, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> ReportBenchWitnessEvidence:
        mapping = _mapping(payload, field_name="ReportBench witness")

        def counts(field_name: str) -> dict[str, int]:
            return {
                key: int(value)
                for key, value in _mapping(
                    mapping.get(field_name) or {}, field_name=field_name
                ).items()
            }

        return cls(
            status=_text(mapping.get("status"), field_name="reportbench.status"),
            witness_role=_text(mapping.get("witness_role"), field_name="reportbench.witness_role"),
            mutation_policy=_text(
                mapping.get("mutation_policy"),
                field_name="reportbench.mutation_policy",
            ),
            source_authority_version=_optional_text(mapping.get("source_authority_version")),
            source_row_ids=_mapping(
                mapping.get("source_row_ids") or {}, field_name="source_row_ids"
            ),
            terminal_truth=_mapping(
                mapping.get("terminal_truth") or {}, field_name="terminal_truth"
            ),
            task_lifecycle_summary=counts("task_lifecycle_summary"),
            validation_summary=counts("validation_summary"),
            evidence_summary=counts("evidence_summary"),
            reportbench_evidence=tuple(
                _mapping_list(
                    mapping.get("reportbench_evidence"),
                    field_name="reportbench_evidence",
                )
            ),
            verifier_verdicts=tuple(
                _mapping_list(
                    mapping.get("verifier_verdicts"),
                    field_name="verifier_verdicts",
                )
            ),
            diagnostics=tuple(
                OperatorEvidenceDiagnostic.from_wire(item)
                for item in list(mapping.get("diagnostics") or [])
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "witness_role": self.witness_role,
            "mutation_policy": self.mutation_policy,
            "source_authority_version": self.source_authority_version,
            "source_row_ids": dict(self.source_row_ids),
            "terminal_truth": dict(self.terminal_truth),
            "task_lifecycle_summary": dict(self.task_lifecycle_summary),
            "validation_summary": dict(self.validation_summary),
            "evidence_summary": dict(self.evidence_summary),
            "reportbench_evidence": [dict(item) for item in self.reportbench_evidence],
            "verifier_verdicts": [dict(item) for item in self.verifier_verdicts],
            "diagnostics": [item.to_wire() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class SmrRunOperatorEvidence:
    schema_version: str
    project_id: str
    run_id: str
    generated_at: datetime
    run_state: str
    projection_freshness: dict[str, Any]
    projection_warnings: tuple[str, ...]
    runtime_timeline: dict[str, Any]
    logical_timeline: dict[str, Any]
    trace_coverage: TraceCoverageEvidence
    transcript_coverage: TranscriptCoverageEvidence
    reportbench_witness: ReportBenchWitnessEvidence
    reconciliation_report: dict[str, Any]
    receipts: tuple[ProjectionEvidenceReceipt, ...]
    diagnostics: tuple[OperatorEvidenceDiagnostic, ...]

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunOperatorEvidence:
        mapping = _mapping(payload, field_name="run operator evidence")
        schema_version = _text(
            mapping.get("schema_version"), field_name="operator_evidence.schema_version"
        )
        if schema_version != "smr_operator_evidence.v1":
            raise ValueError("operator_evidence.schema_version must be smr_operator_evidence.v1")
        return cls(
            schema_version=schema_version,
            project_id=_text(mapping.get("project_id"), field_name="operator_evidence.project_id"),
            run_id=_text(mapping.get("run_id"), field_name="operator_evidence.run_id"),
            generated_at=_datetime(
                mapping.get("generated_at"), field_name="operator_evidence.generated_at"
            ),
            run_state=_text(mapping.get("run_state"), field_name="operator_evidence.run_state"),
            projection_freshness=_mapping(
                mapping.get("projection_freshness") or {},
                field_name="projection_freshness",
            ),
            projection_warnings=_string_tuple(
                mapping.get("projection_warnings"), field_name="projection_warnings"
            ),
            runtime_timeline=_mapping(
                mapping.get("runtime_timeline") or {}, field_name="runtime_timeline"
            ),
            logical_timeline=_mapping(
                mapping.get("logical_timeline") or {}, field_name="logical_timeline"
            ),
            trace_coverage=TraceCoverageEvidence.from_wire(mapping.get("trace_coverage")),
            transcript_coverage=TranscriptCoverageEvidence.from_wire(
                mapping.get("transcript_coverage")
            ),
            reportbench_witness=ReportBenchWitnessEvidence.from_wire(
                mapping.get("reportbench_witness")
            ),
            reconciliation_report=_mapping(
                mapping.get("reconciliation_report") or {},
                field_name="reconciliation_report",
            ),
            receipts=tuple(
                ProjectionEvidenceReceipt.from_wire(item)
                for item in list(mapping.get("receipts") or [])
            ),
            diagnostics=tuple(
                OperatorEvidenceDiagnostic.from_wire(item)
                for item in list(mapping.get("diagnostics") or [])
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        """Serialize exactly the backend ``SmrRunOperatorEvidenceResponse`` DTO."""

        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "run_state": self.run_state,
            "projection_freshness": dict(self.projection_freshness),
            "projection_warnings": list(self.projection_warnings),
            "runtime_timeline": dict(self.runtime_timeline),
            "logical_timeline": dict(self.logical_timeline),
            "trace_coverage": self.trace_coverage.to_wire(),
            "transcript_coverage": self.transcript_coverage.to_wire(),
            "reportbench_witness": self.reportbench_witness.to_wire(),
            "reconciliation_report": dict(self.reconciliation_report),
            "receipts": [item.to_wire() for item in self.receipts],
            "diagnostics": [item.to_wire() for item in self.diagnostics],
        }

    def trust_blockers(self) -> tuple[str, ...]:
        """Return fail-closed blockers for launch-grade operator evidence."""

        blockers: list[str] = []
        blockers.extend(f"projection_warning:{item}" for item in self.projection_warnings)
        all_diagnostics = (
            *self.diagnostics,
            *self.trace_coverage.diagnostics,
            *self.transcript_coverage.diagnostics,
            *self.reportbench_witness.diagnostics,
        )
        blockers.extend(
            f"operator_diagnostic:{diagnostic.code}"
            for diagnostic in all_diagnostics
            if diagnostic.severity.lower() in {"error", "fatal"}
        )
        if self.trace_coverage.missing_trace_count:
            blockers.append("trace_coverage_missing")
        if self.trace_coverage.unverified_trace_count:
            blockers.append("trace_coverage_unverified")
        if self.reportbench_witness.status.lower() != "ok":
            blockers.append("reportbench_witness_not_ok")
        if "read_only_observation_only" not in self.reportbench_witness.mutation_policy:
            blockers.append("reportbench_witness_mutation_policy_invalid")
        for receipt in self.receipts:
            source = receipt.source.lower()
            detail_source_kind = str(receipt.detail.get("source_kind") or "").lower()
            compatibility = receipt.detail.get("compatibility") is True
            if "compatibility" in source or detail_source_kind == "compatibility" or compatibility:
                blockers.append(f"compatibility_receipt:{receipt.receipt_id}")
            if receipt.status.lower() not in {
                "accepted",
                "applied",
                "ok",
                "passed",
                "reconciled",
                "succeeded",
                "verified",
            }:
                blockers.append(f"receipt_not_successful:{receipt.receipt_id}")
        return tuple(dict.fromkeys(blockers))


__all__ = [
    "OperatorEvidenceDiagnostic",
    "ProjectionEvidenceReceipt",
    "ReportBenchWitnessEvidence",
    "SmrRunOperatorEvidence",
    "TraceCoverageEvidence",
    "TranscriptCoverageEvidence",
]
