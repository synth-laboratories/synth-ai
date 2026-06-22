"""Typed local execution profile, local slot contract, and launch helper models."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION = "2026-04-14-local-execution-profile-v2"
LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION = "2026-04-15-local-execution-profile-v3"
LOCAL_EVAL_CONTRACT_SCHEMA_VERSION = "2026-04-14-local-eval-v3"
LOCAL_SOURCE_KIND_SLOT_GIT_MIRROR = "slot_git_mirror"
LOCAL_SOURCE_KIND_EXTERNAL_REPO = "external_repo"
SOURCE_BINDING_KIND_NONE = "none"
SOURCE_BINDING_KIND_TOOL_REPO = "tool_repo"
SOURCE_BINDING_KIND_LOCAL_PRODUCT_SOURCE = "local_product_source"
LOCAL_LAUNCH_TARGET_HOST_KIND = {
    "local-native": "docker",
    "local-docker": "docker",
    "local-dockerized": "docker",
    "daytona": "daytona",
    "local-daytonaized": "daytona",
}
LOCAL_EVAL_CONTRACT_ENV_VARS = (
    "SYNTH_DEV_LOCAL_EVAL_CONTRACT_PATH",
    "SYNTH_EVAL_LOCAL_CONTRACT_PATH",
)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{key} is required")
    return normalized


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _capabilities(payload: Mapping[str, Any]) -> dict[str, bool]:
    raw = payload.get("capabilities")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("capabilities must be an object")
    return {str(key).strip(): bool(value) for key, value in raw.items() if str(key).strip()}


def _repo_ref_from_url(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if "://" in normalized:
        path = normalized.split("://", 1)[1]
        path = path.split("/", 1)[1] if "/" in path else ""
    elif normalized.startswith("git@") and ":" in normalized:
        path = normalized.split(":", 1)[1].strip()
    else:
        path = normalized
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def _repo_ref_from_source_repo(source_repo: Mapping[str, Any] | str | None) -> str:
    if isinstance(source_repo, Mapping):
        return _repo_ref_from_url(str(source_repo.get("url") or ""))
    return _repo_ref_from_url(str(source_repo or ""))


@dataclass(frozen=True)
class LocalExecutionProfile:
    schema_version: str
    profile_id: str
    product: str
    host_kind: str
    docker_image: str | None
    daytona_snapshot: str | None
    required_runtime_kind: str
    source_binding_kind: str
    required_product: str | None
    required_repo: str | None
    local_source_kind: str | None
    capabilities: dict[str, bool]

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> LocalExecutionProfile:
        schema_version = _required_string(payload, "schema_version")
        if schema_version == LEGACY_LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION:
            source_binding_kind = SOURCE_BINDING_KIND_TOOL_REPO
            required_product = None
            required_repo = _required_string(payload, "required_repo")
            local_source_kind = _required_string(payload, "local_source_kind")
        else:
            source_binding_kind = _required_string(payload, "source_binding_kind")
            required_product = _optional_string(payload, "required_product")
            required_repo = _optional_string(payload, "required_repo")
            local_source_kind = _optional_string(payload, "local_source_kind")
        profile = cls(
            schema_version=schema_version,
            profile_id=_required_string(payload, "profile_id"),
            product=_required_string(payload, "product"),
            host_kind=_required_string(payload, "host_kind"),
            docker_image=_optional_string(payload, "docker_image"),
            daytona_snapshot=_optional_string(payload, "daytona_snapshot"),
            required_runtime_kind=_required_string(payload, "required_runtime_kind"),
            source_binding_kind=source_binding_kind,
            required_product=required_product,
            required_repo=required_repo,
            local_source_kind=local_source_kind,
            capabilities=_capabilities(payload),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.schema_version not in {
            LEGACY_LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION,
            LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"unsupported local execution profile schema_version: {self.schema_version}"
            )
        host_kind = self.host_kind.strip().lower()
        if host_kind not in {"docker", "daytona"}:
            raise ValueError(f"unsupported host_kind: {self.host_kind}")
        if host_kind == "docker" and not self.docker_image:
            raise ValueError(f"profile {self.profile_id} requires docker_image")
        if host_kind == "daytona" and not self.daytona_snapshot:
            raise ValueError(f"profile {self.profile_id} requires daytona_snapshot")
        source_binding_kind = self.source_binding_kind.strip().lower()
        if source_binding_kind not in {
            SOURCE_BINDING_KIND_NONE,
            SOURCE_BINDING_KIND_TOOL_REPO,
            SOURCE_BINDING_KIND_LOCAL_PRODUCT_SOURCE,
        }:
            raise ValueError(f"unsupported source_binding_kind: {self.source_binding_kind}")
        local_source_kind = str(self.local_source_kind or "").strip().lower()
        if local_source_kind and local_source_kind not in {
            LOCAL_SOURCE_KIND_SLOT_GIT_MIRROR,
            LOCAL_SOURCE_KIND_EXTERNAL_REPO,
        }:
            raise ValueError(f"unsupported local_source_kind: {self.local_source_kind}")
        required_repo = str(self.required_repo or "").strip()
        required_product = str(self.required_product or "").strip().lower()
        if source_binding_kind == SOURCE_BINDING_KIND_TOOL_REPO and not required_repo:
            raise ValueError(f"profile {self.profile_id} requires required_repo")
        if source_binding_kind == SOURCE_BINDING_KIND_LOCAL_PRODUCT_SOURCE:
            if not required_product or not required_repo:
                raise ValueError(
                    "local_product_source bindings require required_product and required_repo"
                )
            if local_source_kind != LOCAL_SOURCE_KIND_SLOT_GIT_MIRROR:
                raise ValueError(
                    "local_product_source bindings require local_source_kind=slot_git_mirror"
                )

    def to_wire(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "schema_version": self.schema_version,
                "profile_id": self.profile_id,
                "product": self.product,
                "host_kind": self.host_kind,
                "docker_image": self.docker_image,
                "daytona_snapshot": self.daytona_snapshot,
                "required_runtime_kind": self.required_runtime_kind,
                "source_binding_kind": self.source_binding_kind,
                "required_product": self.required_product,
                "required_repo": self.required_repo,
                "local_source_kind": self.local_source_kind,
                "capabilities": dict(self.capabilities),
            }.items()
            if value is not None
        }

    def to_request_wire(self) -> dict[str, Any]:
        return self.to_wire()


@dataclass(frozen=True)
class LocalPublicationReadiness:
    ready: bool
    status: str
    repo: str | None
    credential_name: str | None
    writable_repo_binding_present: bool
    project_connected: bool

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> LocalPublicationReadiness:
        return cls(
            ready=bool(payload.get("ready")),
            status=_required_string(payload, "status"),
            repo=_optional_string(payload, "repo"),
            credential_name=_optional_string(payload, "credential_name"),
            writable_repo_binding_present=bool(payload.get("writable_repo_binding_present")),
            project_connected=bool(payload.get("project_connected")),
        )


@dataclass(frozen=True)
class LocalProductSourceMirror:
    product: str
    source_kind: str
    repo: str
    host_path: str
    runtime_path: str
    default_branch: str | None

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> LocalProductSourceMirror:
        return cls(
            product=_required_string(payload, "product"),
            source_kind=_required_string(payload, "source_kind"),
            repo=_required_string(payload, "repo"),
            host_path=_required_string(payload, "host_path"),
            runtime_path=_required_string(payload, "runtime_path"),
            default_branch=_optional_string(payload, "default_branch"),
        )


@dataclass(frozen=True)
class LocalEvalContract:
    schema_version: str
    slot_id: str
    runtime_id: str
    worker_pool_id: str
    launch_target: str
    requires_hosted_capacity: bool
    product_source_mirrors: dict[str, LocalProductSourceMirror]
    task_env: dict[str, str]

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> LocalEvalContract:
        mirrors_payload = payload.get("product_source_mirrors")
        if mirrors_payload is None:
            mirrors = {}
        elif not isinstance(mirrors_payload, Mapping):
            raise ValueError("product_source_mirrors must be an object")
        else:
            mirrors = {
                str(key).strip().lower(): LocalProductSourceMirror.from_wire(value)
                for key, value in mirrors_payload.items()
                if str(key).strip() and isinstance(value, Mapping)
            }
        raw_task_env = payload.get("task_env")
        task_env = (
            {str(key): str(value) for key, value in raw_task_env.items() if isinstance(key, str)}
            if isinstance(raw_task_env, Mapping)
            else {}
        )
        contract = cls(
            schema_version=_required_string(payload, "schema_version"),
            slot_id=_required_string(payload, "slot_id"),
            runtime_id=_required_string(payload, "runtime_id"),
            worker_pool_id=_required_string(payload, "worker_pool_id"),
            launch_target=_required_string(payload, "launch_target"),
            requires_hosted_capacity=bool(payload.get("requires_hosted_capacity")),
            product_source_mirrors=mirrors,
            task_env=task_env,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.schema_version != LOCAL_EVAL_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported local eval contract schema_version: {self.schema_version}"
            )
        if self.launch_target.strip().lower() not in LOCAL_LAUNCH_TARGET_HOST_KIND:
            raise ValueError(f"unsupported launch_target: {self.launch_target}")


def load_local_execution_profiles(path: Path) -> list[LocalExecutionProfile]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"missing local execution profile manifest: {resolved}")
    payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid local execution profile manifest: {resolved}")
    schema_version = _required_string(payload, "schema_version")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("local execution profile manifest must contain [[profiles]]")
    profiles: list[LocalExecutionProfile] = []
    for item in raw_profiles:
        if not isinstance(item, Mapping):
            raise ValueError("local execution profile entries must be objects")
        normalized = dict(item)
        normalized.setdefault("schema_version", schema_version)
        profiles.append(LocalExecutionProfile.from_wire(normalized))
    if not profiles:
        raise ValueError(f"local execution profile manifest has no profiles: {resolved}")
    return profiles


def load_local_execution_profile(path: Path, *, profile_id: str) -> LocalExecutionProfile:
    requested = str(profile_id or "").strip()
    if not requested:
        raise ValueError("profile_id is required")
    for profile in load_local_execution_profiles(path):
        if profile.profile_id == requested:
            return profile
    raise ValueError(
        f"missing local execution profile {requested!r} in manifest {path.expanduser().resolve()}"
    )


def default_local_eval_contract_path() -> Path | None:
    for env_var in LOCAL_EVAL_CONTRACT_ENV_VARS:
        value = str(os.environ.get(env_var) or "").strip()
        if value:
            return Path(value).expanduser()
    return None


def load_local_eval_contract(path: str | Path | None = None) -> LocalEvalContract:
    resolved = Path(path).expanduser() if path is not None else default_local_eval_contract_path()
    if resolved is None:
        raise ValueError(
            "missing local eval contract path; set SYNTH_DEV_LOCAL_EVAL_CONTRACT_PATH "
            "or SYNTH_EVAL_LOCAL_CONTRACT_PATH"
        )
    material = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(material, Mapping):
        raise ValueError(f"invalid local eval contract payload: {resolved.resolve()}")
    return LocalEvalContract.from_wire(material)


def local_execution_payload(contract: LocalEvalContract) -> dict[str, Any]:
    host_kind = LOCAL_LAUNCH_TARGET_HOST_KIND[contract.launch_target.strip().lower()]
    return {
        "slot_id": contract.slot_id,
        "runtime_id": contract.runtime_id,
        "dispatch_pool": contract.worker_pool_id,
        "host_kind": host_kind,
        "requires_hosted_capacity": contract.requires_hosted_capacity,
    }


def _docker_image_from_contract(contract: LocalEvalContract) -> str:
    for key in (
        "SMR_RUNTIME_IMAGE",
        "LOCAL_SMR_RUNTIME_IMAGE",
        "HORIZONS_RUNTIME_DOCKER_IMAGE",
        "HORIZONS_DOCKER_IMAGE",
    ):
        value = str(contract.task_env.get(key) or "").strip()
        if value:
            return value
    raise ValueError(
        "local eval contract is missing a Docker runtime image for execution_profile.docker_image"
    )


def _daytona_snapshot_from_contract(contract: LocalEvalContract) -> str | None:
    for key in ("HORIZONS_RUNTIME_DAYTONA_SNAPSHOT", "DAYTONA_SNAPSHOT"):
        value = str(contract.task_env.get(key) or "").strip()
        if value:
            return value
    return None


def _mirror_for_repo(
    contract: LocalEvalContract,
    *,
    requested_repo: str,
) -> LocalProductSourceMirror | None:
    normalized_repo = str(requested_repo or "").strip()
    if not normalized_repo:
        return None
    for mirror in contract.product_source_mirrors.values():
        if _repo_ref_from_url(mirror.repo) == normalized_repo:
            return mirror
    return None


def _infer_product(
    contract: LocalEvalContract,
    *,
    requested_repo: str,
    product: str | None,
) -> str:
    explicit = str(product or "").strip().lower()
    if explicit:
        return explicit
    mirror = _mirror_for_repo(contract, requested_repo=requested_repo)
    if mirror is not None:
        return mirror.product.strip().lower()
    return "reportbench"


def local_execution_profile_payload(
    contract: LocalEvalContract,
    *,
    host_kind: str,
    source_repo: Mapping[str, Any] | str | None = None,
    product: str | None = None,
) -> dict[str, Any]:
    normalized_host_kind = str(host_kind or "").strip().lower()
    requested_repo = _repo_ref_from_source_repo(source_repo)
    matched_mirror = _mirror_for_repo(contract, requested_repo=requested_repo)
    resolved_product = _infer_product(
        contract,
        requested_repo=requested_repo,
        product=product,
    )
    profile: dict[str, Any] = {
        "schema_version": LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION,
        "profile_id": f"{resolved_product}_{contract.runtime_id}_{normalized_host_kind}",
        "product": resolved_product,
        "host_kind": normalized_host_kind,
        "required_runtime_kind": contract.runtime_id,
        "capabilities": {},
    }
    if matched_mirror is not None:
        profile["source_binding_kind"] = SOURCE_BINDING_KIND_LOCAL_PRODUCT_SOURCE
        profile["required_product"] = matched_mirror.product
        profile["required_repo"] = matched_mirror.repo
        profile["local_source_kind"] = LOCAL_SOURCE_KIND_SLOT_GIT_MIRROR
    elif requested_repo:
        profile["source_binding_kind"] = SOURCE_BINDING_KIND_TOOL_REPO
        profile["required_repo"] = requested_repo
        profile["local_source_kind"] = LOCAL_SOURCE_KIND_EXTERNAL_REPO
    else:
        profile["source_binding_kind"] = SOURCE_BINDING_KIND_NONE
    if normalized_host_kind == "docker":
        profile["docker_image"] = _docker_image_from_contract(contract)
    elif normalized_host_kind == "daytona":
        snapshot = _daytona_snapshot_from_contract(contract)
        if snapshot:
            profile["daytona_snapshot"] = snapshot
    else:
        raise ValueError(f"unsupported local execution profile host kind: {normalized_host_kind}")
    return profile


def build_local_launch_payload(
    payload: Mapping[str, Any] | None = None,
    *,
    contract: LocalEvalContract,
    host_kind: str | None = None,
    source_repo: Mapping[str, Any] | str | None = None,
    product: str | None = None,
) -> dict[str, Any]:
    local_execution = local_execution_payload(contract)
    expected_host_kind = str(local_execution.get("host_kind") or "").strip().lower()
    requested_host_kind = str(host_kind or "").strip().lower()
    if requested_host_kind and requested_host_kind != expected_host_kind:
        raise ValueError(
            "local eval contract host kind mismatch: "
            f"requested={requested_host_kind} expected={expected_host_kind}"
        )
    base_payload = dict(payload or {})
    base_payload["host_kind"] = expected_host_kind
    base_payload["worker_pool_id"] = contract.worker_pool_id
    base_payload["local_execution"] = dict(local_execution)
    if not isinstance(base_payload.get("execution_profile"), Mapping):
        base_payload["execution_profile"] = local_execution_profile_payload(
            contract,
            host_kind=expected_host_kind,
            source_repo=source_repo,
            product=product,
        )
    return base_payload


__all__ = [
    "LEGACY_LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION",
    "LOCAL_EVAL_CONTRACT_ENV_VARS",
    "LOCAL_EVAL_CONTRACT_SCHEMA_VERSION",
    "LOCAL_EXECUTION_PROFILE_SCHEMA_VERSION",
    "LOCAL_LAUNCH_TARGET_HOST_KIND",
    "LOCAL_SOURCE_KIND_EXTERNAL_REPO",
    "LOCAL_SOURCE_KIND_SLOT_GIT_MIRROR",
    "SOURCE_BINDING_KIND_LOCAL_PRODUCT_SOURCE",
    "SOURCE_BINDING_KIND_NONE",
    "SOURCE_BINDING_KIND_TOOL_REPO",
    "LocalEvalContract",
    "LocalExecutionProfile",
    "LocalProductSourceMirror",
    "LocalPublicationReadiness",
    "build_local_launch_payload",
    "default_local_eval_contract_path",
    "load_local_eval_contract",
    "load_local_execution_profile",
    "load_local_execution_profiles",
    "local_execution_payload",
    "local_execution_profile_payload",
]
