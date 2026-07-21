"""Fail-closed Factory evidence and launch-readiness assembly.

The reducer consumes typed backend owner-route reads. It never reconstructs
state from compatibility projections, local files, database rows, or Redis.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from synth_ai.core.research._legacy.models.factories import (
    Effort,
    Factory,
    FactoryRuntimeStatus,
    FactoryStatus,
)
from synth_ai.core.research._legacy.models.operator_evidence import SmrRunOperatorEvidence
from synth_ai.core.research._legacy.models.project import ManagedResearchProject
from synth_ai.core.research._legacy.models.run_authority import ManagedResearchAuthorityTask
from synth_ai.core.research._legacy.models.run_diagnostics import SmrRunCostSummary
from synth_ai.core.research._legacy.models.run_state import ManagedResearchRun
from synth_ai.core.research._legacy.models.run_timeline import SmrAuthorityReadouts
from synth_ai.core.research._legacy.models.types import RunArtifact
from synth_ai.core.research._legacy.models.work_products import ManagedResearchRunWorkProduct

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CRAFTAX_REPOSITORIES = frozenset({"backend", "evals", "synth-dev", "synth-ai", "gamebench"})
_TERMINAL_RUN_STATES = frozenset({"completed", "failed", "stopped", "cancelled", "archived"})
_SUCCESSFUL_WORK_PRODUCT_STATES = frozenset(
    {"ready", "viewable", "published", "completed", "succeeded"}
)


class FactoryEvidenceValidationError(ValueError):
    """Raised when evidence cannot support the requested claim."""


class FactoryEvidenceTier(StrEnum):
    LOCAL_ACCEPTANCE = "local_acceptance"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class FactoryLaunchTarget(StrEnum):
    LOCAL_ACCEPTANCE = "local_acceptance"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactoryEvidenceValidationError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise FactoryEvidenceValidationError(f"{field_name} keys must be strings")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _required_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    mapping = _mapping(value, field_name=field_name)
    if not mapping:
        raise FactoryEvidenceValidationError(f"{field_name} is required")
    return mapping


def _text(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise FactoryEvidenceValidationError(f"{field_name} is required")
    return normalized


def _number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise FactoryEvidenceValidationError(f"{field_name} must be numeric")
    if not isinstance(value, (int, float, str)):
        raise FactoryEvidenceValidationError(f"{field_name} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FactoryEvidenceValidationError(f"{field_name} must be numeric") from exc


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise FactoryEvidenceValidationError(f"{field_name} must be an integer")
    if not isinstance(value, (int, str)):
        raise FactoryEvidenceValidationError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FactoryEvidenceValidationError(f"{field_name} must be an integer") from exc


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _datetime(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise FactoryEvidenceValidationError(f"{field_name} must be RFC3339") from exc
    else:
        raise FactoryEvidenceValidationError(f"{field_name} must be RFC3339")
    if parsed.tzinfo is None:
        raise FactoryEvidenceValidationError(f"{field_name} must include a timezone")
    return parsed


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FactoryEvidenceValidationError(f"{field_name} must be a list")
    return tuple(
        _text(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _reject_untrusted_markers(value: object, *, field_name: str) -> None:
    """Reject explicit compatibility, reconstructed, or proposed-only markers."""

    if isinstance(value, Mapping):
        mapping = dict(value)
        for key in ("compatibility", "reconstructed", "local_only", "proposed_only"):
            if mapping.get(key) is True:
                raise FactoryEvidenceValidationError(
                    f"{field_name}.{key} evidence is not admissible"
                )
        for key in ("source_kind", "evidence_origin", "evidence_mode"):
            normalized = str(mapping.get(key) or "").strip().lower()
            if normalized in {
                "compatibility",
                "reconstructed",
                "local_only",
                "proposed_only",
            }:
                raise FactoryEvidenceValidationError(
                    f"{field_name}.{key}={normalized} is not admissible"
                )
        reconstructed_from = mapping.get("reconstructed_from")
        if reconstructed_from is not None and reconstructed_from != "" and reconstructed_from != []:
            raise FactoryEvidenceValidationError(
                f"{field_name}.reconstructed_from is not admissible"
            )
        for key, child in mapping.items():
            _reject_untrusted_markers(child, field_name=f"{field_name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_untrusted_markers(child, field_name=f"{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class AppliedExperimentRegistrationReceipt:
    experiment_id: str
    project_id: str
    run_id: str
    intent_id: str
    raw: dict[str, Any]

    @classmethod
    def from_wire(cls, payload: object) -> AppliedExperimentRegistrationReceipt:
        receipt = _mapping(payload, field_name="experiment_registration")
        _reject_untrusted_markers(receipt, field_name="experiment_registration")
        if receipt.get("schema_version") != "craftax_factory.experiment_registration.v1":
            raise FactoryEvidenceValidationError(
                "experiment_registration schema must be craftax_factory.experiment_registration.v1"
            )
        experiment_id = _text(
            receipt.get("experiment_id"), field_name="experiment_registration.experiment_id"
        )
        for field_name in (
            "hypothesis",
            "intervention",
            "comparison",
            "protocol_snapshot",
            "idempotency_key",
        ):
            _text(
                receipt.get(field_name),
                field_name=f"experiment_registration.{field_name}",
            )
        if receipt.get("proposed_before_attempt") is not True:
            raise FactoryEvidenceValidationError(
                "experiment_registration must prove preregistration before the attempt"
            )
        owner = _required_mapping(
            receipt.get("owner_receipt"),
            field_name="experiment_registration.owner_receipt",
        )
        if owner.get("ok") is not True or owner.get("intent_status") != "applied":
            raise FactoryEvidenceValidationError(
                "experiment_registration owner intent must be applied successfully"
            )
        if owner.get("operation") != "experiment.propose":
            raise FactoryEvidenceValidationError(
                "experiment_registration owner operation must be experiment.propose"
            )
        intent_id = _text(
            owner.get("intent_id"),
            field_name="experiment_registration.owner_receipt.intent_id",
        )
        project_id = _text(
            owner.get("project_id"),
            field_name="experiment_registration.owner_receipt.project_id",
        )
        run_id = _text(
            owner.get("run_id"),
            field_name="experiment_registration.owner_receipt.run_id",
        )
        if (
            _text(
                owner.get("experiment_id"),
                field_name="experiment_registration.owner_receipt.experiment_id",
            )
            != experiment_id
        ):
            raise FactoryEvidenceValidationError(
                "experiment_registration owner experiment identity mismatch"
            )
        materialization = _required_mapping(
            owner.get("materialization"),
            field_name="experiment_registration.owner_receipt.materialization",
        )
        if materialization.get("kind") != "experiment_propose":
            raise FactoryEvidenceValidationError(
                "experiment_registration owner materialization is not applied experiment_propose"
            )
        materialized_experiment = _required_mapping(
            materialization.get("experiment"),
            field_name="experiment_registration.owner_receipt.materialization.experiment",
        )
        if materialized_experiment.get("experiment_id") != experiment_id:
            raise FactoryEvidenceValidationError(
                "experiment_registration materialized experiment identity mismatch"
            )
        return cls(
            experiment_id=experiment_id,
            project_id=project_id,
            run_id=run_id,
            intent_id=intent_id,
            raw=receipt,
        )

    def to_wire(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True, slots=True)
class AppliedSynthWikiReceipt:
    changeset_id: str
    proposal_id: str
    experiment_id: str
    project_id: str
    run_id: str
    intent_id: str
    review_state: str
    raw: dict[str, Any]

    @classmethod
    def from_wire(cls, payload: object) -> AppliedSynthWikiReceipt:
        receipt = _mapping(payload, field_name="synth_wiki_changeset")
        _reject_untrusted_markers(receipt, field_name="synth_wiki_changeset")
        if receipt.get("schema_version") != "craftax_factory.synth_wiki_changeset.v1":
            raise FactoryEvidenceValidationError(
                "synth_wiki_changeset schema must be craftax_factory.synth_wiki_changeset.v1"
            )
        changeset_id = _text(
            receipt.get("changeset_id"), field_name="synth_wiki_changeset.changeset_id"
        )
        proposal_id = _text(
            receipt.get("proposal_id"), field_name="synth_wiki_changeset.proposal_id"
        )
        experiment_id = _text(
            receipt.get("source_experiment_id"),
            field_name="synth_wiki_changeset.source_experiment_id",
        )
        run_id = _text(
            receipt.get("source_run_id"),
            field_name="synth_wiki_changeset.source_run_id",
        )
        review_state = _text(
            receipt.get("review_state"), field_name="synth_wiki_changeset.review_state"
        )
        if review_state not in {"proposed", "tentative", "accepted"}:
            raise FactoryEvidenceValidationError("synth_wiki_changeset.review_state is invalid")
        owner = _required_mapping(
            receipt.get("owner_receipt"),
            field_name="synth_wiki_changeset.owner_receipt",
        )
        if owner.get("ok") is not True or owner.get("intent_status") != "applied":
            raise FactoryEvidenceValidationError(
                "synth_wiki_changeset owner intent must be applied successfully"
            )
        if owner.get("operation") != "synth_wiki.propose_edit":
            raise FactoryEvidenceValidationError(
                "synth_wiki_changeset owner operation must be synth_wiki.propose_edit"
            )
        project_id = _text(
            owner.get("project_id"),
            field_name="synth_wiki_changeset.owner_receipt.project_id",
        )
        intent_id = _text(
            owner.get("intent_id"),
            field_name="synth_wiki_changeset.owner_receipt.intent_id",
        )
        expected_owner_values = {
            "run_id": run_id,
            "changeset_id": changeset_id,
            "proposal_id": proposal_id,
            "review_state": review_state,
        }
        for field_name, expected in expected_owner_values.items():
            if str(owner.get(field_name) or "").strip() != expected:
                raise FactoryEvidenceValidationError(
                    f"synth_wiki_changeset owner {field_name} mismatch"
                )
        materialization = _required_mapping(
            owner.get("materialization"),
            field_name="synth_wiki_changeset.owner_receipt.materialization",
        )
        if materialization.get("kind") != "synth_wiki_propose_edit":
            raise FactoryEvidenceValidationError(
                "synth_wiki_changeset materialization must be synth_wiki_propose_edit"
            )
        return cls(
            changeset_id=changeset_id,
            proposal_id=proposal_id,
            experiment_id=experiment_id,
            project_id=project_id,
            run_id=run_id,
            intent_id=intent_id,
            review_state=review_state,
            raw=receipt,
        )

    def to_wire(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True, slots=True)
class ConfirmedProjectGitPushReceipt:
    repository_id: str
    remote_repository: str
    branch: str
    commit_sha: str
    run_id: str
    raw: dict[str, Any]

    @classmethod
    def from_wire(cls, payload: object) -> ConfirmedProjectGitPushReceipt:
        receipt = _mapping(payload, field_name="git_server_receipt")
        _reject_untrusted_markers(receipt, field_name="git_server_receipt")
        if receipt.get("schema_version") != "craftax_factory.git_server_receipt.v1":
            raise FactoryEvidenceValidationError(
                "git_server_receipt schema must be craftax_factory.git_server_receipt.v1"
            )
        repository_id = _text(receipt.get("repo_id"), field_name="git_server_receipt.repo_id")
        remote_repository = _text(
            receipt.get("remote_repo"), field_name="git_server_receipt.remote_repo"
        )
        branch = _text(receipt.get("branch"), field_name="git_server_receipt.branch")
        commit_sha = _text(
            receipt.get("commit_sha"), field_name="git_server_receipt.commit_sha"
        ).lower()
        if not _COMMIT_SHA.fullmatch(commit_sha):
            raise FactoryEvidenceValidationError(
                "git_server_receipt.commit_sha must be 40 lowercase hex characters"
            )
        run_id = _text(receipt.get("source_run_id"), field_name="git_server_receipt.source_run_id")
        if receipt.get("repo_state_advanced") is not True:
            raise FactoryEvidenceValidationError(
                "git_server_receipt must prove repository state advanced"
            )
        owner = _required_mapping(
            receipt.get("workspace_push_receipt"),
            field_name="git_server_receipt.workspace_push_receipt",
        )
        if owner.get("status") != "pushed" or owner.get("git_push_ok") is not True:
            raise FactoryEvidenceValidationError(
                "workspace_push receipt must confirm a successful push"
            )
        if owner.get("archive_only_commit") is not False:
            raise FactoryEvidenceValidationError(
                "archive-only workspace checkpoints are not git-push evidence"
            )
        if owner.get("degraded_success") is not False:
            raise FactoryEvidenceValidationError(
                "degraded workspace_push success is not launch evidence"
            )
        if owner.get("repo_state_advanced") is not True:
            raise FactoryEvidenceValidationError(
                "workspace_push owner receipt must prove repository state advanced"
            )
        if (
            owner.get("had_changes") is not True
            and owner.get("preserved_existing_commit_sha") is not True
        ):
            raise FactoryEvidenceValidationError(
                "workspace_push receipt has no new or preserved commit identity"
            )
        comparisons = {
            "commit_sha": commit_sha,
            "branch": branch,
            "remote_url": remote_repository,
        }
        for field_name, expected in comparisons.items():
            actual = str(owner.get(field_name) or "").strip()
            if field_name == "commit_sha":
                actual = actual.lower()
            if actual != expected:
                raise FactoryEvidenceValidationError(
                    f"workspace_push {field_name} does not match git_server_receipt"
                )
        return cls(
            repository_id=repository_id,
            remote_repository=remote_repository,
            branch=branch,
            commit_sha=commit_sha,
            run_id=run_id,
            raw=receipt,
        )

    def to_wire(self) -> dict[str, Any]:
        return dict(self.raw)


_EXPECTED_BUDGET_POLICY: dict[str, object] = {
    "factory_limit_usd": 180,
    "run_budget_scope": "objective_window",
    "objective_window_days": 30,
    "ordinary_run_limit_usd": 15,
    "ordinary_run_target_usd": 13,
    "tinker_sft_run_limit_usd": 35,
    "max_active_runs": 1,
    "minimum_cycles_funded": 12,
    "tinker_sft_runs_per_window": 1,
    "tinker_requires_operator_approval": True,
}


def _normalized_budget_policy(
    policy: Mapping[str, object], *, cap_policy: Mapping[str, object] | None = None
) -> dict[str, object]:
    normalized = dict(policy)
    if "factory_limit_usd" not in normalized and "limit" in normalized:
        normalized["factory_limit_usd"] = normalized["limit"]
    if "max_active_runs" not in normalized and cap_policy is not None:
        normalized["max_active_runs"] = cap_policy.get("max_active_runs")
    return normalized


@dataclass(frozen=True, slots=True)
class AppliedBudgetOwnerReceipt:
    factory_id: str
    project_id: str
    policy: dict[str, object]
    factory_updated_at: datetime
    project_updated_at: datetime

    @classmethod
    def from_wire(cls, payload: object) -> AppliedBudgetOwnerReceipt:
        receipt = _mapping(payload, field_name="budget_owner_receipt")
        _reject_untrusted_markers(receipt, field_name="budget_owner_receipt")
        if receipt.get("schema_version") != "craftax_factory.budget_owner_receipt.v1":
            raise FactoryEvidenceValidationError(
                "budget_owner_receipt schema must be craftax_factory.budget_owner_receipt.v1"
            )
        if (
            receipt.get("source_authority") != "backend_factory_and_project_owner_routes"
            or receipt.get("applied") is not True
        ):
            raise FactoryEvidenceValidationError(
                "budget_owner_receipt must prove applied backend owner-route state"
            )
        policy = {field_name: receipt.get(field_name) for field_name in _EXPECTED_BUDGET_POLICY}
        for field_name, expected in _EXPECTED_BUDGET_POLICY.items():
            if policy[field_name] != expected:
                raise FactoryEvidenceValidationError(
                    f"budget_owner_receipt {field_name} must equal {expected!r}"
                )
        return cls(
            factory_id=_text(
                receipt.get("factory_id"),
                field_name="budget_owner_receipt.factory_id",
            ),
            project_id=_text(
                receipt.get("project_id"),
                field_name="budget_owner_receipt.project_id",
            ),
            policy=policy,
            factory_updated_at=_datetime(
                receipt.get("factory_updated_at"),
                field_name="budget_owner_receipt.factory_updated_at",
            ),
            project_updated_at=_datetime(
                receipt.get("project_updated_at"),
                field_name="budget_owner_receipt.project_updated_at",
            ),
        )

    @classmethod
    def from_owner_reads(
        cls,
        factory: Factory,
        project: ManagedResearchProject,
    ) -> AppliedBudgetOwnerReceipt:
        if factory.org_id != project.org_id:
            raise FactoryEvidenceValidationError(
                "budget owner reads do not belong to the same organization"
            )
        factory_policy = _normalized_budget_policy(
            factory.budget_policy,
            cap_policy=factory.cap_policy,
        )
        project_policy = _normalized_budget_policy(project.budgets)
        for field_name, expected in _EXPECTED_BUDGET_POLICY.items():
            if factory_policy.get(field_name) != expected:
                raise FactoryEvidenceValidationError(
                    f"factory owner budget policy {field_name} must equal {expected!r}"
                )
            if project_policy.get(field_name) != expected:
                raise FactoryEvidenceValidationError(
                    f"runnable project budget policy {field_name} must equal {expected!r}"
                )
        if factory.updated_at is None or project.updated_at is None:
            raise FactoryEvidenceValidationError(
                "budget owner reads require backend update timestamps"
            )
        return cls(
            factory_id=factory.factory_id,
            project_id=project.project_id,
            policy=dict(_EXPECTED_BUDGET_POLICY),
            factory_updated_at=factory.updated_at,
            project_updated_at=project.updated_at,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": "craftax_factory.budget_owner_receipt.v1",
            "source_authority": "backend_factory_and_project_owner_routes",
            "applied": True,
            "factory_id": self.factory_id,
            "project_id": self.project_id,
            **self.policy,
            "factory_updated_at": self.factory_updated_at.isoformat(),
            "project_updated_at": self.project_updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    components: dict[str, dict[str, object]]
    runtime_image: dict[str, str]
    captured_at: datetime
    manifest_digest: str

    @classmethod
    def from_wire(cls, payload: object) -> SourceIdentity:
        source = _mapping(payload, field_name="source_identity")
        _reject_untrusted_markers(source, field_name="source_identity")
        if set(source) != {"schema_version", "components", "runtime_image", "captured_at"}:
            raise FactoryEvidenceValidationError(
                "source_identity fields must exactly match the Craftax source_identity.v1 contract"
            )
        if source.get("schema_version") != "craftax_factory.source_identity.v1":
            raise FactoryEvidenceValidationError(
                "source_identity.schema_version must be craftax_factory.source_identity.v1"
            )
        raw_components = _mapping(source.get("components"), field_name="source_identity.components")
        if set(raw_components) != _CRAFTAX_REPOSITORIES:
            raise FactoryEvidenceValidationError(
                "source_identity must contain exactly backend, evals, synth-dev, synth-ai, and gamebench"
            )
        components: dict[str, dict[str, object]] = {}
        for repository in sorted(_CRAFTAX_REPOSITORIES):
            component = _mapping(
                raw_components.get(repository),
                field_name=f"source_identity.components.{repository}",
            )
            if set(component) != {"repo", "commit_sha", "source_ref", "dirty"}:
                raise FactoryEvidenceValidationError(
                    f"source_identity.components.{repository} fields do not match the Craftax contract"
                )
            if component.get("repo") != repository:
                raise FactoryEvidenceValidationError(
                    f"source_identity.components.{repository}.repo mismatch"
                )
            commit_sha = str(component.get("commit_sha") or "").strip().lower()
            if not _COMMIT_SHA.fullmatch(commit_sha):
                raise FactoryEvidenceValidationError(
                    f"source_identity.components.{repository}.commit_sha is invalid"
                )
            if component.get("dirty") is not False:
                raise FactoryEvidenceValidationError(
                    f"source_identity.components.{repository}.dirty must be false"
                )
            source_ref = _text(
                component.get("source_ref"),
                field_name=f"source_identity.components.{repository}.source_ref",
            )
            components[repository] = {
                "repo": repository,
                "commit_sha": commit_sha,
                "source_ref": source_ref,
                "dirty": False,
            }
        if components["gamebench"]["commit_sha"] != "5ca3b2b7c99318d0d46db5faaca3f87bf671ea5d":
            raise FactoryEvidenceValidationError(
                "source_identity GameBench commit does not match the frozen Craftax authority"
            )
        raw_runtime_image = _mapping(
            source.get("runtime_image"), field_name="source_identity.runtime_image"
        )
        if set(raw_runtime_image) != {"image_id", "image_ref", "image_digest"}:
            raise FactoryEvidenceValidationError(
                "source_identity.runtime_image fields do not match the Craftax contract"
            )
        runtime_image = {
            "image_id": _text(
                raw_runtime_image.get("image_id"),
                field_name="source_identity.runtime_image.image_id",
            ),
            "image_ref": _text(
                raw_runtime_image.get("image_ref"),
                field_name="source_identity.runtime_image.image_ref",
            ),
            "image_digest": _text(
                raw_runtime_image.get("image_digest"),
                field_name="source_identity.runtime_image.image_digest",
            ).lower(),
        }
        if not _SHA256.fullmatch(runtime_image["image_digest"]):
            raise FactoryEvidenceValidationError(
                "source_identity.runtime_image.image_digest must be sha256:<64 lowercase hex>"
            )
        captured_at_raw = _text(source.get("captured_at"), field_name="source_identity.captured_at")
        try:
            captured_at = datetime.fromisoformat(captured_at_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FactoryEvidenceValidationError(
                "source_identity.captured_at must be RFC3339"
            ) from exc
        if captured_at.tzinfo is None:
            raise FactoryEvidenceValidationError(
                "source_identity.captured_at must include a timezone"
            )
        manifest_material = {
            "components": components,
            "runtime_image": runtime_image,
        }
        material = json.dumps(manifest_material, sort_keys=True, separators=(",", ":"))
        computed_digest = "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        return cls(
            components=components,
            runtime_image=runtime_image,
            captured_at=captured_at,
            manifest_digest=computed_digest,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": "craftax_factory.source_identity.v1",
            "components": {key: dict(value) for key, value in self.components.items()},
            "runtime_image": dict(self.runtime_image),
            "captured_at": self.captured_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeImageIdentity:
    image_reference: str
    image_digest: str
    source_manifest_digest: str

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeImageIdentity:
        image = _mapping(payload, field_name="runtime_image_identity")
        _reject_untrusted_markers(image, field_name="runtime_image_identity")
        image_reference = _text(
            image.get("image_reference") or image.get("image_ref"),
            field_name="runtime_image_identity.image_reference",
        )
        image_digest = _text(
            image.get("image_digest") or image.get("digest"),
            field_name="runtime_image_identity.image_digest",
        ).lower()
        if not _SHA256.fullmatch(image_digest):
            raise FactoryEvidenceValidationError(
                "runtime_image_identity.image_digest must be sha256:<64 lowercase hex>"
            )
        if (
            f"@{image_digest}" not in image_reference
            and image_reference != f"local-docker-image://{image_digest}"
        ):
            raise FactoryEvidenceValidationError(
                "runtime image reference must be an OCI digest or exact local Docker image ID"
            )
        source_manifest_digest = _text(
            image.get("source_manifest_digest"),
            field_name="runtime_image_identity.source_manifest_digest",
        ).lower()
        if not _SHA256.fullmatch(source_manifest_digest):
            raise FactoryEvidenceValidationError(
                "runtime image source_manifest_digest must be sha256:<64 lowercase hex>"
            )
        return cls(
            image_reference=image_reference,
            image_digest=image_digest,
            source_manifest_digest=source_manifest_digest,
        )

    def to_wire(self) -> dict[str, str]:
        return {
            "image_reference": self.image_reference,
            "image_digest": self.image_digest,
            "source_manifest_digest": self.source_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class ActorContainerRunBinding:
    """Verified binding from the launched run to its observed actor container."""

    run_id: str
    slot_id: str
    container_id: str
    container_name: str
    expected_image_id: str
    observed_image_id: str
    observed_at: datetime
    runtime_image_reference: str
    source_identity: SourceIdentity
    raw: dict[str, Any]

    @classmethod
    def from_wire(cls, payload: object) -> ActorContainerRunBinding:
        evidence = _mapping(payload, field_name="runtime_contract_evidence")
        _reject_untrusted_markers(evidence, field_name="runtime_contract_evidence")
        if evidence.get("schema_version") != "craftax_factory.runtime_contract_evidence.v2":
            raise FactoryEvidenceValidationError(
                "runtime_contract_evidence schema must be craftax_factory.runtime_contract_evidence.v2"
            )
        if evidence.get("accepted") is not True or list(evidence.get("acceptance_blockers") or []):
            raise FactoryEvidenceValidationError(
                "runtime_contract_evidence must be accepted with no blockers"
            )
        observations = _required_mapping(
            evidence.get("running_container_observations"),
            field_name="runtime_contract_evidence.running_container_observations",
        )
        actor = _required_mapping(
            observations.get("craftax_actor_runtime"),
            field_name="runtime_contract_evidence.running_container_observations.craftax_actor_runtime",
        )
        if (
            actor.get("schema_version") != "craftax_factory.running_container_image_observation.v1"
            or actor.get("authority") != "synth-dev.local-runtime.docker-inspect"
            or actor.get("role") != "craftax_actor_runtime"
            or actor.get("exact_match") is not True
        ):
            raise FactoryEvidenceValidationError(
                "runtime_contract_evidence actor observation is not an exact synth-dev Docker inspection"
            )
        expected_image_id = _text(
            actor.get("expected_image_id"),
            field_name="actor_container_binding.expected_image_id",
        ).lower()
        observed_image_id = _text(
            actor.get("observed_container_image_id"),
            field_name="actor_container_binding.observed_container_image_id",
        ).lower()
        if not _SHA256.fullmatch(expected_image_id) or observed_image_id != expected_image_id:
            raise FactoryEvidenceValidationError(
                "actor container image ID must exactly match the verified runtime image ID"
            )
        labels = _required_mapping(
            actor.get("container_labels"),
            field_name="actor_container_binding.container_labels",
        )
        if str(labels.get("horizons.managed") or "").strip().lower() != "true":
            raise FactoryEvidenceValidationError(
                "actor container binding must identify a managed Horizons container"
            )
        source_identity = SourceIdentity.from_wire(evidence.get("source_identity"))
        if source_identity.runtime_image["image_id"].lower() != expected_image_id:
            raise FactoryEvidenceValidationError(
                "actor container binding image ID does not match source identity"
            )
        launch_binding = _required_mapping(
            evidence.get("launch_readiness_binding"),
            field_name="runtime_contract_evidence.launch_readiness_binding",
        )
        runtime_image_reference = _text(
            launch_binding.get("smr_runtime_image"),
            field_name="runtime_contract_evidence.launch_readiness_binding.smr_runtime_image",
        )
        if (
            launch_binding.get("exact_match") is not True
            or launch_binding.get("source_identity_runtime_image_ref") != runtime_image_reference
            or source_identity.runtime_image["image_ref"] != runtime_image_reference
        ):
            raise FactoryEvidenceValidationError(
                "runtime contract launch-readiness image binding is not exact"
            )
        evidence_digest = _text(
            evidence.get("evidence_digest"),
            field_name="runtime_contract_evidence.evidence_digest",
        ).lower()
        if not _SHA256.fullmatch(evidence_digest):
            raise FactoryEvidenceValidationError(
                "runtime_contract_evidence.evidence_digest must be sha256:<64 lowercase hex>"
            )
        digest_payload = dict(evidence)
        digest_payload.pop("evidence_digest", None)
        computed_digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(digest_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
        )
        if evidence_digest != computed_digest:
            raise FactoryEvidenceValidationError(
                "runtime_contract_evidence.evidence_digest does not match its payload"
            )
        return cls(
            run_id=_text(
                labels.get("horizons.run_id"),
                field_name="actor_container_binding.horizons.run_id",
            ),
            slot_id=_text(
                labels.get("horizons.slot_id"),
                field_name="actor_container_binding.horizons.slot_id",
            ),
            container_id=_text(
                actor.get("container_id"),
                field_name="actor_container_binding.container_id",
            ),
            container_name=_text(
                actor.get("container_name"),
                field_name="actor_container_binding.container_name",
            ),
            expected_image_id=expected_image_id,
            observed_image_id=observed_image_id,
            observed_at=_datetime(
                actor.get("observed_at"),
                field_name="actor_container_binding.observed_at",
            ),
            runtime_image_reference=runtime_image_reference,
            source_identity=source_identity,
            raw=evidence,
        )

    def to_wire(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True, slots=True)
class RunCostIdentity:
    run_id: str
    total_cents: int
    total_pico_usd: int
    total_usd: float
    recording_status: str
    missing_meters: tuple[str, ...]

    @classmethod
    def from_wire(cls, payload: object) -> RunCostIdentity:
        mapping = _mapping(payload, field_name="cost_identity")
        if mapping.get("source_authority") != "smr_run_cost_summary":
            raise FactoryEvidenceValidationError(
                "cost_identity source_authority must be smr_run_cost_summary"
            )
        recording_status = _text(
            mapping.get("recording_status"),
            field_name="cost_identity.recording_status",
        )
        missing_meters = _string_tuple(
            mapping.get("missing_meters") or (),
            field_name="cost_identity.missing_meters",
        )
        total_cents = _integer(mapping.get("total_cents"), field_name="cost_identity.total_cents")
        total_pico_usd = _integer(
            mapping.get("total_pico_usd"),
            field_name="cost_identity.total_pico_usd",
        )
        total_usd = _number(mapping.get("total_usd"), field_name="cost_identity.total_usd")
        if recording_status != "complete" or missing_meters:
            raise FactoryEvidenceValidationError(
                "cost_identity must have complete recording with no missing meters"
            )
        if total_cents < 0 or total_pico_usd < 0 or total_usd < 0:
            raise FactoryEvidenceValidationError("cost_identity totals must be non-negative")
        return cls(
            run_id=_text(mapping.get("run_id"), field_name="cost_identity.run_id"),
            total_cents=total_cents,
            total_pico_usd=total_pico_usd,
            total_usd=total_usd,
            recording_status=recording_status,
            missing_meters=missing_meters,
        )

    @classmethod
    def from_summary(cls, summary: SmrRunCostSummary) -> RunCostIdentity:
        if summary.recording_status != "complete":
            raise FactoryEvidenceValidationError("run cost recording_status must be complete")
        if summary.missing_meters:
            raise FactoryEvidenceValidationError(
                "run cost identity has missing meters: " + ", ".join(summary.missing_meters)
            )
        if summary.total_cents < 0 or summary.total_pico_usd < 0 or summary.total_usd < 0:
            raise FactoryEvidenceValidationError("run cost totals must be non-negative")
        return cls(
            run_id=summary.run_id,
            total_cents=summary.total_cents,
            total_pico_usd=summary.total_pico_usd,
            total_usd=summary.total_usd,
            recording_status=summary.recording_status,
            missing_meters=summary.missing_meters,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "source_authority": "smr_run_cost_summary",
            "run_id": self.run_id,
            "total_cents": self.total_cents,
            "total_pico_usd": self.total_pico_usd,
            "total_usd": self.total_usd,
            "recording_status": self.recording_status,
            "missing_meters": list(self.missing_meters),
        }


@dataclass(frozen=True, slots=True)
class ArtifactBackedWorkProductIdentity:
    work_product_id: str
    project_id: str
    run_id: str
    kind: str
    title: str
    status: str
    readiness: str
    artifact_id: str
    artifact_uri: str
    artifact_digest: str

    @classmethod
    def from_wire(cls, payload: object) -> ArtifactBackedWorkProductIdentity:
        mapping = _mapping(payload, field_name="artifact-backed WorkProduct")
        artifact = _required_mapping(
            mapping.get("artifact"), field_name="artifact-backed WorkProduct.artifact"
        )
        digest = _text(
            artifact.get("digest"),
            field_name="artifact-backed WorkProduct.artifact.digest",
        ).lower()
        if not _SHA256.fullmatch(digest):
            raise FactoryEvidenceValidationError(
                "artifact-backed WorkProduct digest must be sha256:<64 lowercase hex>"
            )
        status = _text(mapping.get("status"), field_name="WorkProduct.status")
        readiness = _text(mapping.get("readiness"), field_name="WorkProduct.readiness")
        if status.lower() not in _SUCCESSFUL_WORK_PRODUCT_STATES:
            raise FactoryEvidenceValidationError("WorkProduct status is not evidence-ready")
        if readiness.lower() not in _SUCCESSFUL_WORK_PRODUCT_STATES:
            raise FactoryEvidenceValidationError("WorkProduct readiness is not evidence-ready")
        return cls(
            work_product_id=_text(
                mapping.get("work_product_id"), field_name="WorkProduct.work_product_id"
            ),
            project_id=_text(mapping.get("project_id"), field_name="WorkProduct.project_id"),
            run_id=_text(mapping.get("run_id"), field_name="WorkProduct.run_id"),
            kind=_text(mapping.get("kind"), field_name="WorkProduct.kind"),
            title=_text(mapping.get("title"), field_name="WorkProduct.title"),
            status=status,
            readiness=readiness,
            artifact_id=_text(
                artifact.get("artifact_id"), field_name="WorkProduct.artifact.artifact_id"
            ),
            artifact_uri=_text(artifact.get("uri"), field_name="WorkProduct.artifact.uri"),
            artifact_digest=digest,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "work_product_id": self.work_product_id,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "readiness": self.readiness,
            "artifact": {
                "artifact_id": self.artifact_id,
                "uri": self.artifact_uri,
                "digest": self.artifact_digest,
            },
        }


@dataclass(frozen=True, slots=True)
class ArtifactBackedWorkProduct:
    work_product: ManagedResearchRunWorkProduct
    artifact: RunArtifact

    @classmethod
    def from_records(
        cls,
        work_product: ManagedResearchRunWorkProduct,
        artifact: RunArtifact,
    ) -> ArtifactBackedWorkProduct:
        linked_artifact_ids = {link.artifact_id for link in work_product.artifact_links}
        if work_product.artifact_id:
            linked_artifact_ids.add(work_product.artifact_id)
        if not linked_artifact_ids:
            raise FactoryEvidenceValidationError(
                f"WorkProduct {work_product.work_product_id} has no artifact identity"
            )
        if artifact.artifact_id not in linked_artifact_ids:
            raise FactoryEvidenceValidationError(
                f"artifact {artifact.artifact_id} is not linked to WorkProduct {work_product.work_product_id}"
            )
        if artifact.project_id != work_product.project_id or artifact.run_id != work_product.run_id:
            raise FactoryEvidenceValidationError(
                "WorkProduct artifact project/run lineage mismatch"
            )
        if not artifact.digest or not _SHA256.fullmatch(artifact.digest.lower()):
            raise FactoryEvidenceValidationError(
                f"WorkProduct artifact {artifact.artifact_id} lacks an immutable sha256 digest"
            )
        if not artifact.uri:
            raise FactoryEvidenceValidationError(
                f"WorkProduct artifact {artifact.artifact_id} lacks a durable URI"
            )
        if work_product.blocker:
            raise FactoryEvidenceValidationError(
                f"WorkProduct {work_product.work_product_id} has an active blocker"
            )
        if work_product.status.lower() not in _SUCCESSFUL_WORK_PRODUCT_STATES:
            raise FactoryEvidenceValidationError(
                f"WorkProduct {work_product.work_product_id} status is not evidence-ready"
            )
        if work_product.readiness.lower() not in _SUCCESSFUL_WORK_PRODUCT_STATES:
            raise FactoryEvidenceValidationError(
                f"WorkProduct {work_product.work_product_id} readiness is not evidence-ready"
            )
        return cls(work_product=work_product, artifact=artifact)

    def to_wire(self) -> dict[str, Any]:
        payload = {
            "work_product_id": self.work_product.work_product_id,
            "project_id": self.work_product.project_id,
            "run_id": self.work_product.run_id,
            "kind": self.work_product.kind,
            "title": self.work_product.title,
            "status": self.work_product.status,
            "readiness": self.work_product.readiness,
            "artifact": {
                "artifact_id": self.artifact.artifact_id,
                "artifact_type": self.artifact.artifact_type,
                "uri": self.artifact.uri,
                "digest": self.artifact.digest,
                "content_type": self.artifact.content_type,
                "size_bytes": self.artifact.size_bytes,
            },
        }
        ArtifactBackedWorkProductIdentity.from_wire(payload)
        return payload

    def to_identity(self) -> ArtifactBackedWorkProductIdentity:
        return ArtifactBackedWorkProductIdentity.from_wire(self.to_wire())


@dataclass(frozen=True, slots=True)
class FactoryEvidenceReadSet:
    """Typed owner-route inputs for one Factory cycle evidence packet."""

    factory_status: FactoryStatus
    project: ManagedResearchProject
    effort: Effort
    run: ManagedResearchRun
    authority_readouts: SmrAuthorityReadouts
    work_products: tuple[ArtifactBackedWorkProduct, ...]
    cost_identity: RunCostIdentity
    operator_evidence: SmrRunOperatorEvidence

    @property
    def tasks(self) -> tuple[ManagedResearchAuthorityTask, ...]:
        return self.authority_readouts.authority_tasks

    @property
    def execution_turn_ids(self) -> tuple[str, ...]:
        return tuple(turn.execution_turn_id for task in self.tasks for turn in task.execution_turns)


@dataclass(frozen=True, slots=True)
class FactoryEvidencePacket:
    schema_version: str
    evidence_tier: FactoryEvidenceTier
    factory_id: str
    project_id: str
    effort_id: str
    run_id: str
    source_authority_version: str
    task_ids: tuple[str, ...]
    execution_turn_ids: tuple[str, ...]
    work_products: tuple[ArtifactBackedWorkProductIdentity, ...]
    factory_runtime: FactoryRuntimeStatus
    source_identity: SourceIdentity
    runtime_image_identity: RuntimeImageIdentity
    actor_container_binding: ActorContainerRunBinding
    cost_identity: RunCostIdentity
    experiment_registration: AppliedExperimentRegistrationReceipt
    synth_wiki_receipt: AppliedSynthWikiReceipt
    git_push_receipt: ConfirmedProjectGitPushReceipt
    budget_owner_receipt: AppliedBudgetOwnerReceipt
    operator_evidence: SmrRunOperatorEvidence
    assembled_at: datetime

    @classmethod
    def from_wire(cls, payload: object) -> FactoryEvidencePacket:
        mapping = _mapping(payload, field_name="factory_evidence_packet")
        schema_version = _text(
            mapping.get("schema_version"),
            field_name="factory_evidence_packet.schema_version",
        )
        if schema_version != "smr_factory_evidence_packet.v1":
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet schema must be smr_factory_evidence_packet.v1"
            )
        raw_work_products = mapping.get("work_products")
        if not isinstance(raw_work_products, list) or not raw_work_products:
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet.work_products must be a non-empty list"
            )
        packet = cls(
            schema_version=schema_version,
            evidence_tier=FactoryEvidenceTier(
                _text(
                    mapping.get("evidence_tier"),
                    field_name="factory_evidence_packet.evidence_tier",
                )
            ),
            factory_id=_text(
                mapping.get("factory_id"),
                field_name="factory_evidence_packet.factory_id",
            ),
            project_id=_text(
                mapping.get("project_id"),
                field_name="factory_evidence_packet.project_id",
            ),
            effort_id=_text(
                mapping.get("effort_id"),
                field_name="factory_evidence_packet.effort_id",
            ),
            run_id=_text(mapping.get("run_id"), field_name="factory_evidence_packet.run_id"),
            source_authority_version=_text(
                mapping.get("source_authority_version"),
                field_name="factory_evidence_packet.source_authority_version",
            ),
            task_ids=_string_tuple(
                mapping.get("task_ids"),
                field_name="factory_evidence_packet.task_ids",
            ),
            execution_turn_ids=_string_tuple(
                mapping.get("execution_turn_ids"),
                field_name="factory_evidence_packet.execution_turn_ids",
            ),
            work_products=tuple(
                ArtifactBackedWorkProductIdentity.from_wire(item) for item in raw_work_products
            ),
            factory_runtime=FactoryRuntimeStatus.from_wire(mapping.get("factory_runtime")),
            source_identity=SourceIdentity.from_wire(mapping.get("source_identity")),
            runtime_image_identity=RuntimeImageIdentity.from_wire(
                mapping.get("runtime_image_identity")
            ),
            actor_container_binding=ActorContainerRunBinding.from_wire(
                mapping.get("actor_container_binding")
            ),
            cost_identity=RunCostIdentity.from_wire(mapping.get("cost_identity")),
            experiment_registration=AppliedExperimentRegistrationReceipt.from_wire(
                mapping.get("experiment_registration")
            ),
            synth_wiki_receipt=AppliedSynthWikiReceipt.from_wire(mapping.get("synth_wiki_receipt")),
            git_push_receipt=ConfirmedProjectGitPushReceipt.from_wire(
                mapping.get("git_push_receipt")
            ),
            budget_owner_receipt=AppliedBudgetOwnerReceipt.from_wire(
                mapping.get("budget_owner_receipt")
            ),
            operator_evidence=SmrRunOperatorEvidence.from_wire(mapping.get("operator_evidence")),
            assembled_at=_datetime(
                mapping.get("assembled_at"),
                field_name="factory_evidence_packet.assembled_at",
            ),
        )
        if packet.source_identity.manifest_digest != (
            packet.runtime_image_identity.source_manifest_digest
        ):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet source/image manifest mismatch"
            )
        if (
            packet.source_identity.runtime_image["image_ref"]
            != packet.runtime_image_identity.image_reference
            or packet.source_identity.runtime_image["image_digest"]
            != packet.runtime_image_identity.image_digest
        ):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet runtime image identity mismatch"
            )
        if packet.cost_identity.run_id != packet.run_id:
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet cost run identity mismatch"
            )
        if packet.actor_container_binding.run_id != packet.run_id:
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet actor container run identity mismatch"
            )
        if (
            packet.actor_container_binding.runtime_image_reference
            != packet.runtime_image_identity.image_reference
            or packet.actor_container_binding.expected_image_id
            != packet.source_identity.runtime_image["image_id"].lower()
        ):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet actor container image identity mismatch"
            )
        if packet.operator_evidence.project_id != packet.project_id or (
            packet.operator_evidence.run_id != packet.run_id
        ):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet operator evidence identity mismatch"
            )
        if packet.operator_evidence.trust_blockers():
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet operator evidence is not launch-grade"
            )
        if len(set(packet.task_ids)) != len(packet.task_ids) or len(
            set(packet.execution_turn_ids)
        ) != len(packet.execution_turn_ids):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet task/turn identities must be unique"
            )
        if {
            packet.experiment_registration.project_id,
            packet.synth_wiki_receipt.project_id,
            packet.budget_owner_receipt.project_id,
        } != {packet.project_id}:
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet applied receipt project identity mismatch"
            )
        if {
            packet.experiment_registration.run_id,
            packet.synth_wiki_receipt.run_id,
            packet.git_push_receipt.run_id,
        } != {packet.run_id}:
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet applied receipt run identity mismatch"
            )
        if packet.synth_wiki_receipt.experiment_id != (
            packet.experiment_registration.experiment_id
        ):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet Wiki/experiment identity mismatch"
            )
        if packet.budget_owner_receipt.factory_id != packet.factory_id:
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet budget Factory identity mismatch"
            )
        reactor_receipt = packet.factory_runtime.reactor.last_receipt
        if reactor_receipt is None or (
            reactor_receipt.effort_id != packet.effort_id or reactor_receipt.run_id != packet.run_id
        ):
            raise FactoryEvidenceValidationError(
                "factory_evidence_packet reactor identity mismatch"
            )
        for work_product in packet.work_products:
            if work_product.project_id != packet.project_id or (
                work_product.run_id != packet.run_id
            ):
                raise FactoryEvidenceValidationError(
                    "factory_evidence_packet WorkProduct identity mismatch"
                )
        return packet

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_tier": self.evidence_tier.value,
            "factory_id": self.factory_id,
            "project_id": self.project_id,
            "effort_id": self.effort_id,
            "run_id": self.run_id,
            "source_authority_version": self.source_authority_version,
            "task_ids": list(self.task_ids),
            "execution_turn_ids": list(self.execution_turn_ids),
            "work_products": [item.to_wire() for item in self.work_products],
            "factory_runtime": self.factory_runtime.to_wire(),
            "source_identity": self.source_identity.to_wire(),
            "runtime_image_identity": self.runtime_image_identity.to_wire(),
            "actor_container_binding": self.actor_container_binding.to_wire(),
            "cost_identity": self.cost_identity.to_wire(),
            "experiment_registration": self.experiment_registration.to_wire(),
            "synth_wiki_receipt": self.synth_wiki_receipt.to_wire(),
            "git_push_receipt": self.git_push_receipt.to_wire(),
            "budget_owner_receipt": self.budget_owner_receipt.to_wire(),
            "operator_evidence": self.operator_evidence.to_wire(),
            "assembled_at": self.assembled_at.isoformat(),
        }


def assemble_factory_evidence_packet(
    *,
    evidence_tier: FactoryEvidenceTier | str,
    factory: Factory,
    project: ManagedResearchProject,
    effort: Effort,
    run: ManagedResearchRun,
    authority_readouts: SmrAuthorityReadouts,
    factory_runtime: FactoryRuntimeStatus,
    work_products: Sequence[ArtifactBackedWorkProduct],
    source_identity: SourceIdentity,
    runtime_image_identity: RuntimeImageIdentity,
    actor_container_binding: ActorContainerRunBinding,
    cost_identity: RunCostIdentity,
    experiment_registration: AppliedExperimentRegistrationReceipt,
    synth_wiki_receipt: AppliedSynthWikiReceipt,
    git_push_receipt: ConfirmedProjectGitPushReceipt,
    budget_owner_receipt: AppliedBudgetOwnerReceipt,
    operator_evidence: SmrRunOperatorEvidence,
) -> FactoryEvidencePacket:
    """Assemble one cycle from typed owner reads and applied receipts only."""

    tier = (
        evidence_tier
        if isinstance(evidence_tier, FactoryEvidenceTier)
        else FactoryEvidenceTier(str(evidence_tier))
    )
    if len({factory.org_id, project.org_id, effort.org_id}) != 1:
        raise FactoryEvidenceValidationError(
            "Factory, Project, and Effort organization identity mismatch"
        )
    if effort.factory_id != factory.factory_id or effort.project_id != project.project_id:
        raise FactoryEvidenceValidationError("Effort Factory/Project identity mismatch")
    if run.project_id != project.project_id or run.effort_id != effort.effort_id:
        raise FactoryEvidenceValidationError("Run Project/Effort identity mismatch")
    if (
        authority_readouts.project_id != project.project_id
        or authority_readouts.run_id != run.run_id
    ):
        raise FactoryEvidenceValidationError("authority readout Project/Run identity mismatch")
    authority = authority_readouts.typed_runtime_authority
    if authority is None:
        raise FactoryEvidenceValidationError(
            "runtime authority is required; compatibility/public projections are insufficient"
        )
    tasks: tuple[ManagedResearchAuthorityTask, ...] = authority.tasks
    if not tasks:
        raise FactoryEvidenceValidationError("runtime authority contains no typed tasks")
    execution_turn_ids = tuple(
        turn.execution_turn_id for task in tasks for turn in task.execution_turns
    )
    if not execution_turn_ids:
        raise FactoryEvidenceValidationError("runtime authority contains no typed execution turns")
    task_ids = tuple(task.task_id for task in tasks)
    if len(set(task_ids)) != len(task_ids) or len(set(execution_turn_ids)) != len(
        execution_turn_ids
    ):
        raise FactoryEvidenceValidationError(
            "runtime authority contains duplicate task or execution-turn identities"
        )
    if not run.public_state.is_terminal and run.public_state.value not in _TERMINAL_RUN_STATES:
        raise FactoryEvidenceValidationError("run must be terminal before evidence assembly")
    flags = factory_runtime.control_loop_flags
    if not factory_runtime.enabled or factory_runtime.mode != "always_on":
        raise FactoryEvidenceValidationError("Factory runtime must be enabled in always_on mode")
    if not flags.scheduler_enabled or not flags.reactor_enabled:
        raise FactoryEvidenceValidationError(
            "Factory scheduler and reactor owner flags must both be enabled"
        )
    reactor_receipt = factory_runtime.reactor.last_receipt
    if reactor_receipt is None:
        raise FactoryEvidenceValidationError("Factory reactor has no typed terminal-run receipt")
    if reactor_receipt.run_id != run.run_id or reactor_receipt.effort_id != effort.effort_id:
        raise FactoryEvidenceValidationError("Factory reactor receipt Run/Effort identity mismatch")
    if operator_evidence.project_id != project.project_id or operator_evidence.run_id != run.run_id:
        raise FactoryEvidenceValidationError("operator evidence Project/Run identity mismatch")
    trust_blockers = operator_evidence.trust_blockers()
    if trust_blockers:
        raise FactoryEvidenceValidationError(
            "operator evidence is not launch-grade: " + ", ".join(trust_blockers)
        )
    if cost_identity.run_id != run.run_id:
        raise FactoryEvidenceValidationError("cost identity run_id mismatch")
    if source_identity.manifest_digest != runtime_image_identity.source_manifest_digest:
        raise FactoryEvidenceValidationError(
            "runtime image source manifest does not match source identity"
        )
    if (
        source_identity.runtime_image["image_ref"] != runtime_image_identity.image_reference
        or source_identity.runtime_image["image_digest"] != runtime_image_identity.image_digest
    ):
        raise FactoryEvidenceValidationError(
            "runtime image reference/digest does not match source identity"
        )
    if actor_container_binding.run_id != run.run_id:
        raise FactoryEvidenceValidationError("actor container binding run_id mismatch")
    if (
        actor_container_binding.runtime_image_reference != runtime_image_identity.image_reference
        or actor_container_binding.expected_image_id
        != source_identity.runtime_image["image_id"].lower()
        or actor_container_binding.source_identity.to_wire() != source_identity.to_wire()
    ):
        raise FactoryEvidenceValidationError(
            "actor container binding source/runtime image identity mismatch"
        )
    lineage = {
        experiment_registration.project_id,
        synth_wiki_receipt.project_id,
        project.project_id,
        budget_owner_receipt.project_id,
    }
    if len(lineage) != 1:
        raise FactoryEvidenceValidationError("applied receipt Project identity mismatch")
    if {
        experiment_registration.run_id,
        synth_wiki_receipt.run_id,
        git_push_receipt.run_id,
        run.run_id,
    } != {run.run_id}:
        raise FactoryEvidenceValidationError("applied receipt Run identity mismatch")
    if synth_wiki_receipt.experiment_id != experiment_registration.experiment_id:
        raise FactoryEvidenceValidationError(
            "Synth Wiki receipt does not reference the applied experiment"
        )
    if budget_owner_receipt.factory_id != factory.factory_id:
        raise FactoryEvidenceValidationError("budget owner Factory identity mismatch")
    artifact_backed_work_products = tuple(work_products)
    if not artifact_backed_work_products:
        raise FactoryEvidenceValidationError("at least one artifact-backed WorkProduct is required")
    for item in artifact_backed_work_products:
        if (
            item.work_product.project_id != project.project_id
            or item.work_product.run_id != run.run_id
        ):
            raise FactoryEvidenceValidationError(
                "artifact-backed WorkProduct Project/Run identity mismatch"
            )
    return FactoryEvidencePacket(
        schema_version="smr_factory_evidence_packet.v1",
        evidence_tier=tier,
        factory_id=factory.factory_id,
        project_id=project.project_id,
        effort_id=effort.effort_id,
        run_id=run.run_id,
        source_authority_version=authority_readouts.source_authority_version,
        task_ids=task_ids,
        execution_turn_ids=execution_turn_ids,
        work_products=tuple(item.to_identity() for item in artifact_backed_work_products),
        factory_runtime=factory_runtime,
        source_identity=source_identity,
        runtime_image_identity=runtime_image_identity,
        actor_container_binding=actor_container_binding,
        cost_identity=cost_identity,
        experiment_registration=experiment_registration,
        synth_wiki_receipt=synth_wiki_receipt,
        git_push_receipt=git_push_receipt,
        budget_owner_receipt=budget_owner_receipt,
        operator_evidence=operator_evidence,
        assembled_at=_now_utc(),
    )


@dataclass(frozen=True, slots=True)
class FactoryLaunchReadiness:
    schema_version: str
    target: FactoryLaunchTarget
    ready: bool
    blockers: tuple[str, ...]
    evidence_packet: FactoryEvidencePacket
    evaluated_at: datetime

    @classmethod
    def from_wire(cls, payload: object) -> FactoryLaunchReadiness:
        mapping = _mapping(payload, field_name="factory_launch_readiness")
        schema_version = _text(
            mapping.get("schema_version"),
            field_name="factory_launch_readiness.schema_version",
        )
        if schema_version != "smr_factory_launch_readiness.v1":
            raise FactoryEvidenceValidationError(
                "factory_launch_readiness schema must be smr_factory_launch_readiness.v1"
            )
        ready = mapping.get("ready")
        if not isinstance(ready, bool):
            raise FactoryEvidenceValidationError("factory_launch_readiness.ready must be a boolean")
        blockers = _string_tuple(
            mapping.get("blockers") or (),
            field_name="factory_launch_readiness.blockers",
        )
        if ready == bool(blockers):
            raise FactoryEvidenceValidationError(
                "factory_launch_readiness ready/blockers are inconsistent"
            )
        return cls(
            schema_version=schema_version,
            target=FactoryLaunchTarget(
                _text(
                    mapping.get("target"),
                    field_name="factory_launch_readiness.target",
                )
            ),
            ready=ready,
            blockers=blockers,
            evidence_packet=FactoryEvidencePacket.from_wire(mapping.get("evidence_packet")),
            evaluated_at=_datetime(
                mapping.get("evaluated_at"),
                field_name="factory_launch_readiness.evaluated_at",
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target.value,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "evidence_packet": self.evidence_packet.to_wire(),
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    def require_ready(self) -> FactoryLaunchReadiness:
        if not self.ready:
            raise FactoryEvidenceValidationError(
                "Factory launch readiness rejected: " + ", ".join(self.blockers)
            )
        return self


_TIER_ORDER = {
    FactoryEvidenceTier.LOCAL_ACCEPTANCE: 0,
    FactoryEvidenceTier.DEV: 1,
    FactoryEvidenceTier.STAGING: 2,
    FactoryEvidenceTier.PRODUCTION: 3,
}


def assemble_factory_launch_readiness(
    evidence_packet: FactoryEvidencePacket,
    *,
    target: FactoryLaunchTarget | str,
) -> FactoryLaunchReadiness:
    """Assess launch readiness without upgrading local evidence into cloud proof."""

    launch_target = (
        target if isinstance(target, FactoryLaunchTarget) else FactoryLaunchTarget(target)
    )
    target_tier = FactoryEvidenceTier(launch_target.value)
    blockers: list[str] = []
    if _TIER_ORDER[evidence_packet.evidence_tier] < _TIER_ORDER[target_tier]:
        blockers.append(
            f"evidence_tier_{evidence_packet.evidence_tier.value}_cannot_support_{launch_target.value}"
        )
    if launch_target is not FactoryLaunchTarget.LOCAL_ACCEPTANCE and (
        evidence_packet.evidence_tier is FactoryEvidenceTier.LOCAL_ACCEPTANCE
    ):
        blockers.append("local_only_evidence_rejected_for_cloud_launch")
    return FactoryLaunchReadiness(
        schema_version="smr_factory_launch_readiness.v1",
        target=launch_target,
        ready=not blockers,
        blockers=tuple(blockers),
        evidence_packet=evidence_packet,
        evaluated_at=_now_utc(),
    )


__all__ = [
    "ActorContainerRunBinding",
    "AppliedBudgetOwnerReceipt",
    "AppliedExperimentRegistrationReceipt",
    "AppliedSynthWikiReceipt",
    "ArtifactBackedWorkProduct",
    "ArtifactBackedWorkProductIdentity",
    "ConfirmedProjectGitPushReceipt",
    "FactoryEvidencePacket",
    "FactoryEvidenceReadSet",
    "FactoryEvidenceTier",
    "FactoryEvidenceValidationError",
    "FactoryLaunchReadiness",
    "FactoryLaunchTarget",
    "RunCostIdentity",
    "RuntimeImageIdentity",
    "SourceIdentity",
    "assemble_factory_evidence_packet",
    "assemble_factory_launch_readiness",
]
