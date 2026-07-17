"""Strict public models for isolated GameBench candidate scoring."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ENTRYPOINT = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_IDENTIFIER_FIELDS = (
    "request_id",
    "idempotency_key",
    "project_id",
    "run_id",
    "work_product_id",
    "candidate_id",
    "execution_contract_version",
    "task_id",
    "suite_id",
    "policy_identity",
    "deployment_id",
    "claim_id",
)
_SOURCE_FIELDS = ("gamebench_source_sha", "scorer_source_sha")
_SHA256_FIELDS = (
    "candidate_sha256",
    "scorer_fixture_manifest_sha256",
    "scorer_binary_sha256",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty bounded canonical string")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


def _require_sha1(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA1.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GameBenchCandidateScoreIdentity(_StrictModel):
    """Every authority field echoed through the complete score-job lifecycle."""

    request_id: str
    idempotency_key: str
    project_id: str
    run_id: str
    work_product_id: str
    candidate_id: str
    candidate_sha256: str
    execution_contract_version: str
    entrypoint: str
    task_id: str
    suite_id: str
    seeds: tuple[int, ...]
    max_steps: int
    lane: Literal["rust"]
    policy_identity: str
    gamebench_source_sha: str
    scorer_source_sha: str
    scorer_fixture_manifest_sha256: str
    scorer_binary_sha256: str
    scorer_image_digest: str
    network_mode: Literal["none"]
    environment_allowlist: tuple[str, ...]
    timeout_seconds: float
    environment: Literal["dev", "staging", "prod"]
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"]
    deployment_id: str
    claim_id: str
    fencing_token: int

    @field_validator("seeds", "environment_allowlist", mode="before")
    @classmethod
    def _json_arrays_to_tuples(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_identity(self) -> GameBenchCandidateScoreIdentity:
        for field in _IDENTIFIER_FIELDS:
            _require_identifier(getattr(self, field), field)
        if (
            not self.entrypoint
            or self.entrypoint != self.entrypoint.strip()
            or _SAFE_ENTRYPOINT.fullmatch(self.entrypoint) is None
            or self.entrypoint.startswith("/")
            or ".." in self.entrypoint.split("/")
        ):
            raise ValueError("entrypoint must be a safe canonical relative path")
        for field in _SOURCE_FIELDS:
            _require_sha1(getattr(self, field), field)
        for field in _SHA256_FIELDS:
            _require_sha256(getattr(self, field), field)
        if _IMAGE_DIGEST.fullmatch(self.scorer_image_digest) is None:
            raise ValueError("scorer_image_digest must be an immutable sha256 digest")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique and preserve the requested order")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2_147_483_647
            for seed in self.seeds
        ):
            raise ValueError("seeds must contain only non-negative signed 32-bit integers")
        if isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.environment_allowlist != ():
            raise ValueError("environment_allowlist must be explicitly empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(self.fencing_token, bool) or self.fencing_token <= 0:
            raise ValueError("fencing_token must be a positive integer")
        return self

    def authority_fields(self) -> dict[str, Any]:
        """Return the exact identity fields used for response equality checks."""
        return self.model_dump(mode="python")


class GameBenchCandidateScoreRequest(GameBenchCandidateScoreIdentity):
    """Complete request for one immutable candidate and one frozen Craftax suite."""

    candidate_bytes: bytes = Field(repr=False)

    @model_validator(mode="after")
    def _validate_candidate_bytes(self) -> GameBenchCandidateScoreRequest:
        if not self.candidate_bytes:
            raise ValueError("candidate_bytes must be non-empty")
        if b"\x00" in self.candidate_bytes:
            raise ValueError("candidate_bytes must not contain NUL")
        try:
            self.candidate_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("candidate_bytes must be strict UTF-8") from exc
        digest = hashlib.sha256(self.candidate_bytes).hexdigest()
        if digest != self.candidate_sha256:
            raise ValueError("candidate_bytes do not match candidate_sha256")
        return self

    def authority_fields(self) -> dict[str, Any]:
        """Return identity fields without materializing candidate source in logs."""
        payload = super().authority_fields()
        payload.pop("candidate_bytes", None)
        return payload

    def to_wire(self) -> dict[str, Any]:
        """Encode candidate bytes once, without an alternate text-codec path."""
        payload = self.authority_fields()
        payload["candidate_bytes"] = base64.b64encode(self.candidate_bytes).decode("ascii")
        payload["seeds"] = list(self.seeds)
        payload["environment_allowlist"] = []
        return payload

    def request_body_sha256(self) -> str:
        """Hash the exact canonical wire body used for idempotency binding."""
        return _canonical_json_sha256(self.to_wire())

    @classmethod
    def from_wire(cls, payload: Any) -> GameBenchCandidateScoreRequest:
        """Parse the one accepted JSON representation for candidate bytes."""
        if not isinstance(payload, dict):
            raise TypeError("candidate score request must be a JSON object")
        raw = payload.get("candidate_bytes")
        if not isinstance(raw, str):
            raise TypeError("candidate_bytes wire value must be base64 text")
        try:
            candidate_bytes = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("candidate_bytes wire value is not canonical base64") from exc
        if base64.b64encode(candidate_bytes).decode("ascii") != raw:
            raise ValueError("candidate_bytes wire value is not canonical base64")
        decoded = dict(payload)
        decoded["candidate_bytes"] = candidate_bytes
        return cls.model_validate(decoded)


class GameBenchCandidateScoreInputAuthority(_StrictModel):
    """Server-bankable proof of the accepted immutable request body."""

    schema_version: Literal["gamebench.craftax.score_input_authority.v1"]
    request_body_sha256: str
    candidate_sha256: str
    candidate_size_bytes: int
    idempotency_key: str

    @model_validator(mode="after")
    def _validate_input_authority(self) -> GameBenchCandidateScoreInputAuthority:
        _require_sha256(self.request_body_sha256, "request_body_sha256")
        _require_sha256(self.candidate_sha256, "candidate_sha256")
        _require_identifier(self.idempotency_key, "idempotency_key")
        if isinstance(self.candidate_size_bytes, bool) or self.candidate_size_bytes <= 0:
            raise ValueError("candidate_size_bytes must be positive")
        return self


class GameBenchCandidateScoreSubmission(GameBenchCandidateScoreIdentity):
    """Accepted score job with every request identity echoed by the service."""

    schema_version: Literal["gamebench.craftax.score_submission.v1"]
    job_id: str
    status: Literal["accepted"]
    created_at: str
    input_authority: GameBenchCandidateScoreInputAuthority

    @model_validator(mode="after")
    def _validate_submission(self) -> GameBenchCandidateScoreSubmission:
        _require_identifier(self.job_id, "job_id")
        _require_identifier(self.created_at, "created_at")
        if self.input_authority.candidate_sha256 != self.candidate_sha256:
            raise ValueError("submission input authority candidate SHA mismatch")
        if self.input_authority.idempotency_key != self.idempotency_key:
            raise ValueError("submission input authority idempotency mismatch")
        return self


class GameBenchCandidateScoreRow(_StrictModel):
    """One ordered, independently attributable held-out seed score."""

    seed: int
    reward: float
    achievement_count: int
    achievements: tuple[str, ...]
    rollout_id: str
    policy_isolation: Literal["os_sandbox_observation_action.v2"]
    episode_status: Literal["succeeded"]

    @field_validator("achievements", mode="before")
    @classmethod
    def _achievements_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_row(self) -> GameBenchCandidateScoreRow:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("score row seed must be non-negative")
        if not math.isfinite(self.reward):
            raise ValueError("score row reward must be finite")
        if isinstance(self.achievement_count, bool) or self.achievement_count < 0:
            raise ValueError("achievement_count must be non-negative")
        if self.achievement_count != len(self.achievements):
            raise ValueError("achievement_count does not match achievements")
        if tuple(sorted(set(self.achievements))) != self.achievements:
            raise ValueError("achievements must be sorted and unique")
        _require_identifier(self.rollout_id, "rollout_id")
        return self


class GameBenchCandidateScoreOutputAuthority(_StrictModel):
    """Typed provenance and isolation receipt for terminal score output."""

    schema_version: Literal["gamebench.craftax.score_output_authority.v1"]
    score_rows_sha256: str
    candidate_sha256: str
    gamebench_source_sha: str
    scorer_source_sha: str
    scorer_fixture_manifest_sha256: str
    scorer_binary_sha256: str
    scorer_image_digest: str
    task_id: str
    suite_id: str
    seeds: tuple[int, ...]
    max_steps: int
    lane: Literal["rust"]
    policy_identity: str
    isolation_contract: Literal["os_sandbox_observation_action.v2"]
    network_mode: Literal["none"]
    environment_allowlist: tuple[str, ...]
    candidate_suite_visible: Literal[False]
    candidate_evaluator_visible: Literal[False]
    terminal: Literal[True]
    environment: Literal["dev", "staging", "prod"]
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"]
    deployment_id: str
    claim_id: str
    fencing_token: int

    @field_validator("seeds", "environment_allowlist", mode="before")
    @classmethod
    def _output_arrays_to_tuples(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_output_authority(self) -> GameBenchCandidateScoreOutputAuthority:
        _require_sha256(self.score_rows_sha256, "score_rows_sha256")
        _require_sha256(self.candidate_sha256, "candidate_sha256")
        _require_sha1(self.gamebench_source_sha, "gamebench_source_sha")
        _require_sha1(self.scorer_source_sha, "scorer_source_sha")
        _require_sha256(
            self.scorer_fixture_manifest_sha256,
            "scorer_fixture_manifest_sha256",
        )
        _require_sha256(self.scorer_binary_sha256, "scorer_binary_sha256")
        if _IMAGE_DIGEST.fullmatch(self.scorer_image_digest) is None:
            raise ValueError("scorer_image_digest must be immutable")
        if self.environment_allowlist != ():
            raise ValueError("output environment_allowlist must be explicitly empty")
        _require_identifier(self.deployment_id, "deployment_id")
        _require_identifier(self.claim_id, "claim_id")
        if isinstance(self.fencing_token, bool) or self.fencing_token <= 0:
            raise ValueError("output fencing_token must be positive")
        return self


class GameBenchCandidateScoreFailure(_StrictModel):
    """Trusted failure classification that never includes candidate diagnostics."""

    code: str
    origin: Literal["candidate", "trusted_supervisor", "trusted_scorer"]
    retryable: bool

    @model_validator(mode="after")
    def _validate_failure(self) -> GameBenchCandidateScoreFailure:
        _require_identifier(self.code, "failure.code")
        return self


class GameBenchCandidateScoreResult(GameBenchCandidateScoreIdentity):
    """Terminal score result returned by ``client.gamebench.scorers.wait``."""

    schema_version: Literal["gamebench.craftax.score_result.v1"]
    job_id: str
    result_id: str
    status: Literal["succeeded", "failed", "timed_out", "cancelled"]
    mean_reward: float | None
    benchmark_score: float | None
    score_rows: tuple[GameBenchCandidateScoreRow, ...]
    output_authority: GameBenchCandidateScoreOutputAuthority | None
    failure: GameBenchCandidateScoreFailure | None
    created_at: str
    started_at: str | None
    finished_at: str

    @field_validator("score_rows", mode="before")
    @classmethod
    def _rows_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_result(self) -> GameBenchCandidateScoreResult:
        for field in ("job_id", "result_id", "created_at", "finished_at"):
            _require_identifier(getattr(self, field), field)
        if self.status == "succeeded":
            if self.started_at is None:
                raise ValueError("succeeded score result requires started_at")
            _require_identifier(self.started_at, "started_at")
            if self.mean_reward is None or self.benchmark_score is None:
                raise ValueError("succeeded score result requires both aggregate scores")
            if not math.isfinite(self.mean_reward) or not math.isfinite(self.benchmark_score):
                raise ValueError("succeeded score aggregates must be finite")
            if self.failure is not None or self.output_authority is None:
                raise ValueError("succeeded score result requires authority and no failure")
            if tuple(row.seed for row in self.score_rows) != self.seeds:
                raise ValueError("score rows do not preserve the exact requested seed order")
            rows_wire = [row.model_dump(mode="json") for row in self.score_rows]
            if _canonical_json_sha256(rows_wire) != self.output_authority.score_rows_sha256:
                raise ValueError("score rows do not match output authority")
            expected = {
                "candidate_sha256": self.candidate_sha256,
                "gamebench_source_sha": self.gamebench_source_sha,
                "scorer_source_sha": self.scorer_source_sha,
                "scorer_fixture_manifest_sha256": self.scorer_fixture_manifest_sha256,
                "scorer_binary_sha256": self.scorer_binary_sha256,
                "scorer_image_digest": self.scorer_image_digest,
                "task_id": self.task_id,
                "suite_id": self.suite_id,
                "seeds": self.seeds,
                "max_steps": self.max_steps,
                "lane": self.lane,
                "policy_identity": self.policy_identity,
                "network_mode": self.network_mode,
                "environment_allowlist": self.environment_allowlist,
                "environment": self.environment,
                "cloud_slot": self.cloud_slot,
                "deployment_id": self.deployment_id,
                "claim_id": self.claim_id,
                "fencing_token": self.fencing_token,
            }
            actual = self.output_authority.model_dump(mode="python")
            for key, value in expected.items():
                if actual[key] != value:
                    raise ValueError(f"result output authority mismatch for {key}")
        else:
            if self.mean_reward is not None or self.benchmark_score is not None:
                raise ValueError("non-succeeded result must not carry aggregate scores")
            if self.score_rows or self.output_authority is not None or self.failure is None:
                raise ValueError("non-succeeded result must carry only a typed failure")
        return self


class GameBenchCandidateScoreCancellation(GameBenchCandidateScoreIdentity):
    """Typed cancellation result for one exact score job."""

    schema_version: Literal["gamebench.craftax.score_cancellation.v1"]
    job_id: str
    result_id: str | None
    status: Literal["cancelled", "already_terminal"]
    process_group_terminated: bool
    cleanup_required: bool
    cancelled_at: str

    @model_validator(mode="after")
    def _validate_cancellation(self) -> GameBenchCandidateScoreCancellation:
        _require_identifier(self.job_id, "job_id")
        _require_identifier(self.cancelled_at, "cancelled_at")
        if self.result_id is not None:
            _require_identifier(self.result_id, "result_id")
        if self.status == "cancelled" and not self.process_group_terminated:
            raise ValueError("cancelled job must prove process-group termination")
        return self


class GameBenchCandidateScoreCleanupReceipt(GameBenchCandidateScoreIdentity):
    """Terminal proof that no job-owned runtime resources remain."""

    schema_version: Literal["gamebench.craftax.score_cleanup.v1"]
    job_id: str
    result_id: str | None
    status: Literal["cleaned"]
    workspace_removed: bool
    process_group_terminated: bool
    policy_sandboxes_removed: bool
    processes_remaining: int
    containers_remaining: int
    resources_remaining: int
    state_record_disposition: Literal["retained_until_external_archive"]
    cleaned_at: str

    @model_validator(mode="after")
    def _validate_cleanup(self) -> GameBenchCandidateScoreCleanupReceipt:
        _require_identifier(self.job_id, "job_id")
        _require_identifier(self.cleaned_at, "cleaned_at")
        if self.result_id is not None:
            _require_identifier(self.result_id, "result_id")
        counts = (
            self.processes_remaining,
            self.containers_remaining,
            self.resources_remaining,
        )
        if any(isinstance(value, bool) or value != 0 for value in counts):
            raise ValueError("cleaned receipt must prove zero remaining resources")
        if not (
            self.workspace_removed
            and self.process_group_terminated
            and self.policy_sandboxes_removed
        ):
            raise ValueError("cleaned receipt must prove every cleanup dimension")
        return self


def score_rows_sha256(rows: tuple[GameBenchCandidateScoreRow, ...]) -> str:
    """Hash score rows using the canonical wire representation."""
    return _canonical_json_sha256([row.model_dump(mode="json") for row in rows])


__all__ = [
    "GameBenchCandidateScoreCancellation",
    "GameBenchCandidateScoreCleanupReceipt",
    "GameBenchCandidateScoreFailure",
    "GameBenchCandidateScoreIdentity",
    "GameBenchCandidateScoreInputAuthority",
    "GameBenchCandidateScoreOutputAuthority",
    "GameBenchCandidateScoreRequest",
    "GameBenchCandidateScoreResult",
    "GameBenchCandidateScoreRow",
    "GameBenchCandidateScoreSubmission",
    "score_rows_sha256",
]
