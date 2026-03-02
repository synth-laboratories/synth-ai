"""Managed Research control-plane SDK client.

This module exposes a focused client for controlling Synth Managed Research (SMR)
projects and runs through public API routes.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import httpx

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.container.auth import encrypt_for_backend

ACTIVE_RUN_STATES = {"queued", "planning", "executing", "blocked", "finalizing", "running"}
DEFAULT_TIMEOUT_SECONDS = 30.0
_FUNDING_SOURCE_ALIASES = {"byok": "synth", "customer": "synth"}
_VALID_FUNDING_SOURCES = {"synth"}

__all__ = [
    "ACTIVE_RUN_STATES",
    "DEFAULT_TIMEOUT_SECONDS",
    "ManagedResearchClient",
    "SmrActorStatus",
    "SmrApiError",
    "SmrApproval",
    "SmrArtifact",
    "SmrCapabilities",
    "SmrControlClient",
    "SmrProject",
    "SmrProjectStatusSnapshot",
    "SmrQuestion",
    "SmrRun",
    "SmrRunEconomics",
    "SmrRunLogArchive",
    "first_id",
]


class SmrApiError(RuntimeError):
    """Raised when the SMR API returns a non-success response."""


def _resolve_backend_base(backend_base: str | None) -> str:
    candidate = (backend_base or os.getenv("SYNTH_BACKEND_URL") or BACKEND_URL_BASE).strip()
    if not candidate:
        candidate = "https://api.usesynth.ai"
    return normalize_backend_base(candidate).rstrip("/")


def _resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key
    try:
        resolved = get_api_key("SYNTH_API_KEY", required=True)
    except Exception:
        resolved = os.environ.get("SYNTH_API_KEY", "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide api_key or set SYNTH_API_KEY)")
    return resolved


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _normalize_provider_funding_source(funding_source: str) -> str:
    normalized = (funding_source or "").strip().lower()
    normalized = _FUNDING_SOURCE_ALIASES.get(normalized, normalized)
    if normalized not in _VALID_FUNDING_SOURCES:
        raise ValueError(f"funding_source must be one of {sorted(_VALID_FUNDING_SOURCES)}")
    return normalized


def _coerce_list(data: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in (
            "items",
            "data",
            "results",
            "projects",
            "runs",
            "questions",
            "approvals",
            "artifacts",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise SmrApiError(f"Expected list response for {label}, received {type(data).__name__}")


def _coerce_dict(data: Any, *, label: str) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    raise SmrApiError(f"Expected object response for {label}, received {type(data).__name__}")


@dataclass(frozen=True)
class SmrProject:
    project_id: str
    org_id: str | None
    name: str | None
    archived: bool | None
    created_at: str | None
    updated_at: str | None
    onboarding_state: dict[str, Any]
    execution: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrProject:
        return cls(
            project_id=str(payload.get("project_id") or ""),
            org_id=(str(payload.get("org_id")) if payload.get("org_id") is not None else None),
            name=(str(payload.get("name")) if payload.get("name") is not None else None),
            archived=(
                bool(payload.get("archived")) if payload.get("archived") is not None else None
            ),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") is not None else None
            ),
            updated_at=(
                str(payload.get("updated_at")) if payload.get("updated_at") is not None else None
            ),
            onboarding_state=dict(payload.get("onboarding_state") or {}),
            execution=dict(payload.get("execution") or {}),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrRun:
    run_id: str
    org_id: str | None
    project_id: str | None
    trigger: str | None
    state: str | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    status_detail: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrRun:
        return cls(
            run_id=str(payload.get("run_id") or ""),
            org_id=(str(payload.get("org_id")) if payload.get("org_id") is not None else None),
            project_id=(
                str(payload.get("project_id")) if payload.get("project_id") is not None else None
            ),
            trigger=(str(payload.get("trigger")) if payload.get("trigger") is not None else None),
            state=(str(payload.get("state")) if payload.get("state") is not None else None),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") is not None else None
            ),
            started_at=(
                str(payload.get("started_at")) if payload.get("started_at") is not None else None
            ),
            finished_at=(
                str(payload.get("finished_at")) if payload.get("finished_at") is not None else None
            ),
            status_detail=dict(payload.get("status_detail") or {}),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrQuestion:
    question_id: str
    run_id: str
    project_id: str
    status: str
    prompt: str
    metadata: dict[str, Any]
    response_text: str | None
    created_at: str | None
    responded_at: str | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrQuestion:
        return cls(
            question_id=str(payload.get("question_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            project_id=str(payload.get("project_id") or ""),
            status=str(payload.get("status") or ""),
            prompt=str(payload.get("prompt") or ""),
            metadata=dict(payload.get("metadata") or {}),
            response_text=(
                str(payload.get("response_text"))
                if payload.get("response_text") is not None
                else None
            ),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") is not None else None
            ),
            responded_at=(
                str(payload.get("responded_at"))
                if payload.get("responded_at") is not None
                else None
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrApproval:
    approval_id: str
    run_id: str
    project_id: str
    kind: str
    status: str
    title: str | None
    body: str | None
    metadata: dict[str, Any]
    created_at: str | None
    resolved_at: str | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrApproval:
        return cls(
            approval_id=str(payload.get("approval_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            project_id=str(payload.get("project_id") or ""),
            kind=str(payload.get("kind") or ""),
            status=str(payload.get("status") or ""),
            title=(str(payload.get("title")) if payload.get("title") is not None else None),
            body=(str(payload.get("body")) if payload.get("body") is not None else None),
            metadata=dict(payload.get("metadata") or {}),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") is not None else None
            ),
            resolved_at=(
                str(payload.get("resolved_at")) if payload.get("resolved_at") is not None else None
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrArtifact:
    artifact_id: str
    run_id: str
    project_id: str
    artifact_type: str
    title: str | None
    uri: str | None
    digest: str | None
    metadata: dict[str, Any]
    created_at: str | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrArtifact:
        return cls(
            artifact_id=str(payload.get("artifact_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            project_id=str(payload.get("project_id") or ""),
            artifact_type=str(payload.get("artifact_type") or ""),
            title=(str(payload.get("title")) if payload.get("title") is not None else None),
            uri=(str(payload.get("uri")) if payload.get("uri") is not None else None),
            digest=(str(payload.get("digest")) if payload.get("digest") is not None else None),
            metadata=dict(payload.get("metadata") or {}),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") is not None else None
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrActorStatus:
    actor_id: str
    actor_type: str
    project_id: str
    run_id: str
    state: str
    phase: str | None
    task_id: str | None
    task_key: str | None
    updated_at: str | None
    last_heartbeat_at: str | None
    paused_at: str | None
    error_summary: str | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrActorStatus:
        return cls(
            actor_id=str(payload.get("actor_id") or ""),
            actor_type=str(payload.get("actor_type") or ""),
            project_id=str(payload.get("project_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            state=str(payload.get("state") or ""),
            phase=(str(payload.get("phase")) if payload.get("phase") is not None else None),
            task_id=(str(payload.get("task_id")) if payload.get("task_id") is not None else None),
            task_key=(
                str(payload.get("task_key")) if payload.get("task_key") is not None else None
            ),
            updated_at=(
                str(payload.get("updated_at")) if payload.get("updated_at") is not None else None
            ),
            last_heartbeat_at=(
                str(payload.get("last_heartbeat_at"))
                if payload.get("last_heartbeat_at") is not None
                else None
            ),
            paused_at=(
                str(payload.get("paused_at")) if payload.get("paused_at") is not None else None
            ),
            error_summary=(
                str(payload.get("error_summary"))
                if payload.get("error_summary") is not None
                else None
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrRunLogArchive:
    log_archive_id: str
    run_id: str
    project_id: str
    storage_backend: str | None
    record_count: int
    session_count: int
    created_at: str | None
    metadata: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrRunLogArchive:
        return cls(
            log_archive_id=str(payload.get("log_archive_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            project_id=str(payload.get("project_id") or ""),
            storage_backend=(
                str(payload.get("storage_backend"))
                if payload.get("storage_backend") is not None
                else None
            ),
            record_count=int(payload.get("record_count") or 0),
            session_count=int(payload.get("session_count") or 0),
            created_at=(
                str(payload.get("created_at")) if payload.get("created_at") is not None else None
            ),
            metadata=dict(payload.get("metadata") or {}),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrRunEconomics:
    run_id: str
    org_id: str | None
    project_id: str | None
    summary: dict[str, Any]
    spend_entries: list[dict[str, Any]]
    egress_events: list[dict[str, Any]]
    trace_artifact: dict[str, Any] | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrRunEconomics:
        return cls(
            run_id=str(payload.get("run_id") or ""),
            org_id=(str(payload.get("org_id")) if payload.get("org_id") is not None else None),
            project_id=(
                str(payload.get("project_id")) if payload.get("project_id") is not None else None
            ),
            summary=dict(payload.get("summary") or {}),
            spend_entries=list(payload.get("spend_entries") or []),
            egress_events=list(payload.get("egress_events") or []),
            trace_artifact=(
                dict(payload.get("trace_artifact"))
                if isinstance(payload.get("trace_artifact"), dict)
                else None
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrCapabilities:
    supports_project_scoped_runs: bool
    supports_run_list_filters: bool
    supports_run_list_cursor: bool
    supports_project_status_snapshot: bool
    supports_unified_actor_status: bool
    actor_status_schema_version: str | None
    supports_actor_control: bool
    supports_encrypted_provider_keys: bool
    supports_run_economics: bool
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrCapabilities:
        return cls(
            supports_project_scoped_runs=bool(payload.get("supports_project_scoped_runs")),
            supports_run_list_filters=bool(payload.get("supports_run_list_filters")),
            supports_run_list_cursor=bool(payload.get("supports_run_list_cursor")),
            supports_project_status_snapshot=bool(payload.get("supports_project_status_snapshot")),
            supports_unified_actor_status=bool(payload.get("supports_unified_actor_status")),
            actor_status_schema_version=(
                str(payload.get("actor_status_schema_version"))
                if payload.get("actor_status_schema_version") is not None
                else None
            ),
            supports_actor_control=bool(payload.get("supports_actor_control")),
            supports_encrypted_provider_keys=bool(payload.get("supports_encrypted_provider_keys")),
            supports_run_economics=bool(payload.get("supports_run_economics")),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class SmrProjectStatusSnapshot:
    project_id: str
    state: str | None
    active_run_id: str | None
    active_run_state: str | None
    active_runs: list[dict[str, Any]]
    active_actor_summaries: list[dict[str, Any]]
    queue_backlog_counts: dict[str, int]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmrProjectStatusSnapshot:
        queue_counts_raw = payload.get("queue_backlog_counts")
        queue_counts: dict[str, int] = {}
        if isinstance(queue_counts_raw, dict):
            queue_counts = {
                str(key): int(value)
                for key, value in queue_counts_raw.items()
                if key is not None and value is not None
            }
        return cls(
            project_id=str(payload.get("project_id") or ""),
            state=(str(payload.get("state")) if payload.get("state") is not None else None),
            active_run_id=(
                str(payload.get("active_run_id"))
                if payload.get("active_run_id") is not None
                else None
            ),
            active_run_state=(
                str(payload.get("active_run_state"))
                if payload.get("active_run_state") is not None
                else None
            ),
            active_runs=[
                item for item in list(payload.get("active_runs") or []) if isinstance(item, dict)
            ],
            active_actor_summaries=[
                item
                for item in list(payload.get("active_actor_summaries") or [])
                if isinstance(item, dict)
            ],
            queue_backlog_counts=queue_counts,
            raw=dict(payload),
        )


@dataclass
class SmrControlClient:
    """SMR control-plane client with compatibility stricts."""

    api_key: str | None = None
    backend_base: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_api_key = _resolve_api_key(self.api_key)
        resolved_backend_base = _resolve_backend_base(self.backend_base)
        self.api_key = resolved_api_key
        self.backend_base = resolved_backend_base
        self._client = httpx.Client(
            base_url=resolved_backend_base,
            headers=_auth_headers(resolved_api_key),
            timeout=self.timeout_seconds,
        )

    def __enter__(self) -> SmrControlClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        response = self._client.request(method.upper(), path, params=params, json=json_body)
        if allow_not_found and response.status_code == 404:
            return None
        if response.status_code >= 400:
            snippet = response.text[:500] if response.text else ""
            raise SmrApiError(f"{method.upper()} {path} failed ({response.status_code}): {snippet}")
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    # Project lifecycle -------------------------------------------------

    def create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", "/smr/projects", json_body=payload)

    def list_projects(
        self,
        *,
        include_archived: bool = False,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "include_archived": int(include_archived),
            "limit": int(limit),
        }
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        data = self._request_json(
            "GET",
            "/smr/projects",
            params=params,
        )
        return _coerce_list(data, label="list_projects")

    def list_projects_typed(
        self,
        *,
        include_archived: bool = False,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[SmrProject]:
        return [
            SmrProject.from_dict(item)
            for item in self.list_projects(
                include_archived=include_archived,
                created_after=created_after,
                created_before=created_before,
                limit=limit,
                cursor=cursor,
            )
        ]

    def get_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/projects/{project_id}")

    def get_project_typed(self, project_id: str) -> SmrProject:
        return SmrProject.from_dict(_coerce_dict(self.get_project(project_id), label="get_project"))

    def patch_project(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("PATCH", f"/smr/projects/{project_id}", json_body=payload)

    def get_project_status(self, project_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/projects/{project_id}/status")

    def get_project_status_snapshot(self, project_id: str) -> dict[str, Any]:
        """Polling-friendly project status snapshot."""
        return self.get_project_status(project_id)

    def get_project_status_snapshot_typed(self, project_id: str) -> SmrProjectStatusSnapshot:
        payload = _coerce_dict(
            self.get_project_status_snapshot(project_id), label="get_project_status_snapshot"
        )
        return SmrProjectStatusSnapshot.from_dict(payload)

    def get_project_entitlement(self, project_id: str) -> dict[str, Any]:
        return self._request_json(
            "GET", f"/smr/projects/{project_id}/entitlements/managed_research"
        )

    def get_capabilities(self) -> dict[str, Any]:
        return self._request_json("GET", "/smr/capabilities")

    def get_capabilities_typed(self) -> SmrCapabilities:
        return SmrCapabilities.from_dict(
            _coerce_dict(self.get_capabilities(), label="get_capabilities")
        )

    def pause_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/projects/{project_id}/pause")

    def resume_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/projects/{project_id}/resume")

    def archive_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/projects/{project_id}/archive")

    def unarchive_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/projects/{project_id}/unarchive")

    # Onboarding --------------------------------------------------------

    def onboarding_start(self, project_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/projects/{project_id}/onboarding/start")

    def onboarding_complete_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark an onboarding step complete or skipped.

        Step keys used by the wizard: ``connect_github``, ``starting_data_spec``,
        ``keys_budgets``. Backend also supports ``artifact_storage``,
        ``approvals_egress``, ``connect_linear``, ``notifications``.
        """
        payload: dict[str, Any] = {"step": step, "status": status}
        if detail is not None:
            payload["detail"] = detail
        return self._request_json(
            "POST",
            f"/smr/projects/{project_id}/onboarding/complete_step",
            json_body=payload,
        )

    def onboarding_dry_run(self, project_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/projects/{project_id}/onboarding/dry_run")

    def onboarding_status(self, project_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/projects/{project_id}/onboarding/status")

    # Provider keys -----------------------------------------------------

    def set_provider_key(
        self,
        project_id: str,
        *,
        provider: str,
        funding_source: str,
        api_key: str | None = None,
        encrypted_key_b64: str | None = None,
        encrypt_before_send: bool = False,
    ) -> dict[str, Any]:
        provider_api_key = api_key
        if not provider_api_key and not encrypted_key_b64:
            raise ValueError("api_key or encrypted_key_b64 is required")
        funding_source_norm = _normalize_provider_funding_source(funding_source)

        payload: dict[str, Any] = {
            "provider": provider,
            "funding_source": funding_source_norm,
        }

        if encrypted_key_b64:
            payload["encrypted_key_b64"] = encrypted_key_b64
        elif encrypt_before_send and provider_api_key:
            pub = self._request_json("GET", "/api/v1/crypto/public-key")
            public_key = str((pub or {}).get("public_key") or "").strip()
            if not public_key:
                raise SmrApiError("Backend did not return /api/v1/crypto/public-key public_key")
            payload["encrypted_key_b64"] = encrypt_for_backend(public_key, provider_api_key)
        else:
            payload["api_key"] = provider_api_key

        response = self._client.post(f"/smr/projects/{project_id}/provider_keys", json=payload)

        if response.status_code >= 400:
            snippet = response.text[:500] if response.text else ""
            raise SmrApiError(
                f"POST /smr/projects/{project_id}/provider_keys failed "
                f"({response.status_code}): {snippet}"
            )
        return response.json() if response.content else {}

    def provider_key_status(
        self, project_id: str, provider: str, funding_source: str
    ) -> dict[str, Any]:
        funding_source_norm = _normalize_provider_funding_source(funding_source)
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/provider_keys/{provider}/{funding_source_norm}/status",
        )

    # Starting data ----------------------------------------------------

    def get_starting_data_upload_urls(
        self,
        project_id: str,
        *,
        files: list[dict[str, Any]],
        dataset_ref: str | None = None,
    ) -> dict[str, Any]:
        if not files:
            raise ValueError("files must contain at least one entry")

        payload_files: list[dict[str, str]] = []
        for file in files:
            if not isinstance(file, dict):
                raise ValueError("each file entry must be a JSON object")
            path = str(file.get("path") or "").strip()
            if not path:
                raise ValueError("each file entry requires non-empty 'path'")
            entry: dict[str, str] = {"path": path}
            content_type = file.get("content_type")
            if isinstance(content_type, str) and content_type.strip():
                entry["content_type"] = content_type.strip()
            payload_files.append(entry)

        resolved_dataset_ref = (
            dataset_ref.strip()
            if isinstance(dataset_ref, str) and dataset_ref.strip()
            else "starting-data"
        )
        payload: dict[str, Any] = {
            "dataset_ref": resolved_dataset_ref,
            "files": payload_files,
        }

        return self._request_json(
            "POST",
            f"/smr/projects/{project_id}/starting-data/upload-urls",
            json_body=payload,
        )

    def upload_starting_data_files(
        self,
        project_id: str,
        *,
        files: list[dict[str, Any]],
        dataset_ref: str | None = None,
    ) -> dict[str, Any]:
        if not files:
            raise ValueError("files must contain at least one entry")

        request_files: list[dict[str, str]] = []
        file_payloads: dict[str, tuple[bytes | Path, str | None]] = {}
        seen_paths: set[str] = set()
        for file in files:
            if not isinstance(file, dict):
                raise ValueError("each file entry must be a JSON object")
            path = str(file.get("path") or "").strip()
            if not path:
                raise ValueError("each file entry requires non-empty 'path'")
            if path in seen_paths:
                raise ValueError(f"duplicate file path in upload payload: '{path}'")
            seen_paths.add(path)

            content = file.get("content")
            content_path = file.get("content_path")
            if content is not None and content_path is not None:
                raise ValueError(f"file '{path}' cannot set both 'content' and 'content_path'")
            payload_source: bytes | Path
            if content_path is not None:
                if not isinstance(content_path, (str, os.PathLike)):
                    raise ValueError(f"file '{path}' requires 'content_path' as str|Path")
                payload_path = Path(content_path).expanduser()
                if not payload_path.exists():
                    raise ValueError(f"file '{path}' content_path does not exist: {payload_path}")
                if not payload_path.is_file():
                    raise ValueError(f"file '{path}' content_path is not a file: {payload_path}")
                payload_source = payload_path
            elif isinstance(content, str):
                payload_source = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                payload_source = bytes(content)
            else:
                raise ValueError(f"file '{path}' requires 'content' as str|bytes or 'content_path'")

            entry: dict[str, str] = {"path": path}
            content_type = file.get("content_type")
            resolved_content_type: str | None = None
            if isinstance(content_type, str) and content_type.strip():
                resolved_content_type = content_type.strip()
                entry["content_type"] = resolved_content_type

            request_files.append(entry)
            file_payloads[path] = (payload_source, resolved_content_type)

        upload_response = self.get_starting_data_upload_urls(
            project_id,
            files=request_files,
            dataset_ref=dataset_ref,
        )

        uploads = upload_response.get("uploads")
        if not isinstance(uploads, list):
            raise SmrApiError("starting-data upload response missing 'uploads' list")

        with httpx.Client(timeout=self.timeout_seconds) as upload_client:
            for upload in uploads:
                if not isinstance(upload, dict):
                    raise SmrApiError("starting-data upload response contains invalid upload item")
                path = str(upload.get("path") or "").strip()
                upload_url = str(upload.get("upload_url") or "").strip()
                if not path or not upload_url:
                    raise SmrApiError("starting-data upload item missing path or upload_url")
                payload = file_payloads.get(path)
                if payload is None:
                    raise SmrApiError(
                        f"starting-data upload response returned unknown path '{path}'"
                    )
                payload_source, content_type = payload
                content_bytes = (
                    payload_source.read_bytes()
                    if isinstance(payload_source, Path)
                    else payload_source
                )
                headers: dict[str, str] = {}
                if content_type:
                    headers["Content-Type"] = content_type
                response = upload_client.put(
                    upload_url, content=content_bytes, headers=headers or None
                )
                if response.status_code >= 400:
                    snippet = response.text[:500] if response.text else ""
                    raise SmrApiError(
                        f"PUT starting-data upload for '{path}' failed "
                        f"({response.status_code}): {snippet}"
                    )

        return upload_response

    def upload_starting_data_directory(
        self,
        project_id: str,
        directory: str | os.PathLike[str],
        *,
        dataset_ref: str | None = None,
    ) -> dict[str, Any]:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"directory does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"path is not a directory: {root}")

        files_to_upload: list[dict[str, Any]] = []
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            rel_path = file_path.relative_to(root).as_posix()
            content_type, _ = mimetypes.guess_type(file_path.name)
            files_to_upload.append(
                {
                    "path": rel_path,
                    "content_path": file_path,
                    "content_type": content_type or "application/octet-stream",
                }
            )
        if not files_to_upload:
            raise ValueError(f"directory has no files to upload: {root}")

        return self.upload_starting_data_files(
            project_id,
            files=files_to_upload,
            dataset_ref=dataset_ref,
        )

    # Runs --------------------------------------------------------------

    def trigger_run(
        self,
        project_id: str,
        *,
        timebox_seconds: int | None = None,
        agent_model: str | None = None,
        agent_kind: str | None = None,
        workflow: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Trigger a run, optionally overriding agent model and kind for this run only.

        Args:
            project_id: Project to trigger.
            timebox_seconds: Optional run time limit in seconds (minimum 60).
            agent_model: Override model for this run, e.g. ``"claude-opus-4-5"`` or
                ``"gpt-4o"``.  Persisted in ``status_detail.agent_model_override`` and
                applied by the orchestrator at claim time.
            agent_kind: Override agent runtime for this run: ``"codex"``, ``"claude"``,
                or ``"opencode"``.  Persisted in ``status_detail.agent_kind_override``.
            workflow: Optional workflow payload for specialized run rails (for
                example ``{"kind": "data_factory_v1", ...}``). When omitted,
                run behavior is unchanged.
        """
        payload: dict[str, Any] = {}
        if timebox_seconds is not None:
            payload["timebox_seconds"] = int(timebox_seconds)
        if agent_model and agent_model.strip():
            payload["agent_model"] = agent_model.strip()
        if agent_kind and agent_kind.strip():
            payload["agent_kind"] = agent_kind.strip().lower()
        if workflow is not None:
            payload["workflow"] = workflow
        return self._request_json("POST", f"/smr/projects/{project_id}/trigger", json_body=payload)

    def trigger_data_factory_run(
        self,
        project_id: str,
        *,
        dataset_ref: str,
        bundle_manifest_path: str,
        profile: str = "founder_default",
        source_mode: str = "mcp_local",
        targets: list[str] | None = None,
        preferred_target: str = "harbor",
        runtime_kind: str | None = None,
        environment_kind: str | None = None,
        strictness_mode: str = "warn",
        timebox_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Trigger a standardized Data Factory workflow run."""
        workflow_targets = targets[:] if targets else [preferred_target or "harbor"]
        workflow_payload: dict[str, Any] = {
            "kind": "data_factory_v1",
            "profile": profile,
            "source_mode": source_mode,
            "targets": workflow_targets,
            "preferred_target": preferred_target,
            "input": {
                "dataset_ref": dataset_ref,
                "bundle_manifest_path": bundle_manifest_path,
            },
            "options": {
                "strictness_mode": strictness_mode,
            },
        }
        if runtime_kind:
            workflow_payload["runtime_kind"] = runtime_kind
        if environment_kind:
            workflow_payload["environment_kind"] = environment_kind
        return self.trigger_run(
            project_id,
            timebox_seconds=timebox_seconds,
            workflow=workflow_payload,
        )

    # -- Data Factory dedicated API ------------------------------------------

    def data_factory_finalize(
        self,
        project_id: str,
        *,
        dataset_ref: str = "starting-data",
        bundle_manifest_path: str = "capture_bundle.json",
        target_formats: list[str] | None = None,
        preferred_target: str = "harbor",
        finalizer_profile: str = "founder_default",
        source_mode: str = "mcp_local",
        runtime_kind: str | None = None,
        environment_kind: str | None = None,
        strictness_mode: str = "warn",
        timebox_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Submit a Data Factory finalization job via the dedicated API."""
        payload: dict[str, Any] = {
            "dataset_ref": dataset_ref,
            "bundle_manifest_path": bundle_manifest_path,
            "target_formats": target_formats or [preferred_target or "harbor"],
            "preferred_target": preferred_target,
            "finalizer_profile": finalizer_profile,
            "source_mode": source_mode,
            "strictness_mode": strictness_mode,
        }
        if runtime_kind:
            payload["runtime_kind"] = runtime_kind
        if environment_kind:
            payload["environment_kind"] = environment_kind
        if timebox_seconds is not None:
            payload["timebox_seconds"] = int(timebox_seconds)
        return self._request_json(
            "POST",
            f"/smr/projects/{project_id}/data-factory/finalize",
            json_body=payload,
        )

    def data_factory_finalize_status(
        self,
        project_id: str,
        job_id: str,
    ) -> dict[str, Any]:
        """Get the status of a Data Factory finalization job."""
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/data-factory/finalize/{job_id}",
        )

    def data_factory_publish(
        self,
        project_id: str,
        job_id: str,
        *,
        reason: str = "manual_publish",
    ) -> dict[str, Any]:
        """Publish finalized Data Factory artifacts."""
        return self._request_json(
            "POST",
            f"/smr/projects/{project_id}/data-factory/finalize/{job_id}/publish",
            json_body={"reason": reason},
        )

    def set_agent_config(
        self,
        project_id: str,
        *,
        model: str | None = None,
        agent_kind: str | None = None,
    ) -> dict[str, Any]:
        """Set the default agent model and/or kind for all future runs of a project.

        Writes into ``project.execution.agent_model`` and/or
        ``project.execution.agent_kind``.  Existing keys in ``execution`` are preserved
        (uses a read-patch-write to avoid clobbering other execution config).

        Args:
            project_id: Project to configure.
            model: Model string, e.g. ``"claude-opus-4-5"``, ``"gpt-4o"``,
                ``"claude-haiku-4-5-20251001"``.
            agent_kind: Agent runtime: ``"codex"`` (default), ``"claude"``, or
                ``"opencode"``.
        """
        if model is None and agent_kind is None:
            raise ValueError("at least one of model or agent_kind is required")
        project = self.get_project(project_id)
        execution: dict[str, Any] = dict(project.get("execution") or {})
        if model is not None:
            execution["agent_model"] = model.strip()
        if agent_kind is not None:
            ak = agent_kind.strip().lower()
            if ak not in {"codex", "claude", "opencode"}:
                raise ValueError("agent_kind must be 'codex', 'claude', or 'opencode'")
            execution["agent_kind"] = ak
        return self._request_json(
            "PATCH",
            f"/smr/projects/{project_id}",
            json_body={"execution": execution},
        )

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        scoped = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs",
            allow_not_found=True,
        )
        if scoped is not None:
            return _coerce_list(scoped, label="project_scoped_list_runs")
        canonical = self._request_json(
            "GET",
            "/smr/runs",
            params={"project_id": project_id},
            allow_not_found=True,
        )
        if canonical is None:
            return []
        return _coerce_list(canonical, label="canonical_list_runs")

    def list_runs_typed(self, project_id: str) -> list[SmrRun]:
        return [SmrRun.from_dict(item) for item in self.list_runs(project_id)]

    def list_project_runs(
        self,
        project_id: str,
        *,
        state: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        """List runs for a project with optional query filters."""
        params: dict[str, Any] = {}
        if state:
            params["state"] = state
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        params["limit"] = int(limit)
        if cursor:
            params["cursor"] = cursor
        data = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs",
            params=params or None,
        )
        return _coerce_list(data, label="list_project_runs")

    def list_project_runs_typed(
        self,
        project_id: str,
        *,
        state: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> list[SmrRun]:
        rows = self.list_project_runs(
            project_id,
            state=state,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            cursor=cursor,
        )
        return [SmrRun.from_dict(item) for item in rows]

    def list_active_runs(self, project_id: str) -> list[dict[str, Any]]:
        scoped = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/active",
            allow_not_found=True,
        )
        if scoped is not None:
            return _coerce_list(scoped, label="project_scoped_list_active_runs")
        runs = self.list_runs(project_id)
        out: list[dict[str, Any]] = []
        for run in runs:
            state = str(run.get("state") or "").strip().lower()
            if state in ACTIVE_RUN_STATES:
                out.append(run)
        return out

    def get_actor_status(self, project_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        params = {"run_id": run_id} if run_id else None
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/actors/status",
            params=params,
        )

    def get_actor_status_typed(
        self, project_id: str, *, run_id: str | None = None
    ) -> list[SmrActorStatus]:
        payload = _coerce_dict(
            self.get_actor_status(project_id, run_id=run_id), label="get_actor_status"
        )
        actor_rows = payload.get("actors")
        if not isinstance(actor_rows, list):
            raise SmrApiError("Expected get_actor_status response to include an 'actors' list")
        return [SmrActorStatus.from_dict(row) for row in actor_rows if isinstance(row, dict)]

    def get_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}",
                allow_not_found=True,
            )
            if scoped is not None:
                return scoped
        return self._request_json("GET", f"/smr/runs/{run_id}")

    def get_run_typed(self, run_id: str, *, project_id: str | None = None) -> SmrRun:
        return SmrRun.from_dict(
            _coerce_dict(self.get_run(run_id, project_id=project_id), label="get_run")
        )

    def pause_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/runs/{run_id}/pause")

    def resume_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/runs/{run_id}/resume")

    def create_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if checkpoint_id is not None:
            payload["checkpoint_id"] = checkpoint_id
        if reason is not None:
            payload["reason"] = reason
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/checkpoints",
                json_body=payload or {},
                allow_not_found=True,
            )
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/checkpoints",
            json_body=payload or {},
        )

    def list_run_checkpoints(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/checkpoints",
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_checkpoints")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/checkpoints")
        return _coerce_list(canonical, label="canonical_list_run_checkpoints")

    def restore_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if checkpoint_id is not None:
            payload["checkpoint_id"] = checkpoint_id
        if reason is not None:
            payload["reason"] = reason
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/restore",
                json_body=payload or {},
                allow_not_found=True,
            )
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/restore",
            json_body=payload or {},
        )

    def stop_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/runs/{run_id}/stop")

    def control_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        action: Literal["pause", "resume"],
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action}
        if reason is not None:
            payload["reason"] = reason
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        return self._request_json(
            "POST",
            f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_id}/control",
            json_body=payload,
        )

    def pause_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.control_actor(
            project_id,
            run_id,
            actor_id,
            action="pause",
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def resume_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.control_actor(
            project_id,
            run_id,
            actor_id,
            action="resume",
            reason=reason,
            idempotency_key=idempotency_key,
        )

    # Questions + approvals --------------------------------------------

    def list_project_questions(
        self,
        project_id: str,
        *,
        status_filter: str = "pending",
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "status_filter": status_filter,
            "limit": int(limit),
        }
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        data = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/questions",
            params=params,
        )
        return _coerce_list(data, label="list_project_questions")

    def list_run_questions(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if status_filter:
            params["status_filter"] = status_filter
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/questions",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_questions")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/questions", params=params)
        return _coerce_list(canonical, label="canonical_list_run_questions")

    def list_run_questions_typed(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[SmrQuestion]:
        return [
            SmrQuestion.from_dict(item)
            for item in self.list_run_questions(
                run_id,
                project_id=project_id,
                status_filter=status_filter,
                created_after=created_after,
                created_before=created_before,
                limit=limit,
                cursor=cursor,
            )
        ]

    def respond_question(
        self,
        run_id: str,
        question_id: str,
        *,
        response_text: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {"response_text": response_text}
        if project_id:
            scoped_path = (
                f"/smr/projects/{project_id}/runs/{run_id}/questions/{question_id}/respond"
            )
            scoped = self._request_json(
                "POST", scoped_path, json_body=payload, allow_not_found=True
            )
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/questions/{question_id}/respond",
            json_body=payload,
        )

    def list_project_approvals(
        self,
        project_id: str,
        *,
        status_filter: str = "pending",
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "status_filter": status_filter,
            "limit": int(limit),
        }
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        data = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/approvals",
            params=params,
        )
        return _coerce_list(data, label="list_project_approvals")

    def list_run_approvals(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if status_filter:
            params["status_filter"] = status_filter
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_approvals")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/approvals", params=params)
        return _coerce_list(canonical, label="canonical_list_run_approvals")

    def list_run_approvals_typed(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[SmrApproval]:
        return [
            SmrApproval.from_dict(item)
            for item in self.list_run_approvals(
                run_id,
                project_id=project_id,
                status_filter=status_filter,
                created_after=created_after,
                created_before=created_before,
                limit=limit,
                cursor=cursor,
            )
        ]

    def approve(
        self,
        run_id: str,
        approval_id: str,
        *,
        comment: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if comment is not None:
            payload["comment"] = comment
        if project_id:
            scoped_path = (
                f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/approve"
            )
            scoped = self._request_json(
                "POST", scoped_path, json_body=payload, allow_not_found=True
            )
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/approvals/{approval_id}/approve",
            json_body=payload,
        )

    def deny(
        self,
        run_id: str,
        approval_id: str,
        *,
        comment: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if comment is not None:
            payload["comment"] = comment
        if project_id:
            scoped_path = f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/deny"
            scoped = self._request_json(
                "POST", scoped_path, json_body=payload, allow_not_found=True
            )
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/approvals/{approval_id}/deny",
            json_body=payload,
        )

    # Artifacts ---------------------------------------------------------

    def list_run_artifacts(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        artifact_type: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if artifact_type:
            params["artifact_type"] = artifact_type
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before
        if cursor:
            params["cursor"] = cursor
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/artifacts",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_artifacts")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/artifacts", params=params)
        return _coerce_list(canonical, label="canonical_list_run_artifacts")

    def list_run_artifacts_typed(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        artifact_type: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[SmrArtifact]:
        return [
            SmrArtifact.from_dict(item)
            for item in self.list_run_artifacts(
                run_id,
                project_id=project_id,
                artifact_type=artifact_type,
                created_after=created_after,
                created_before=created_before,
                limit=limit,
                cursor=cursor,
            )
        ]

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/artifacts/{artifact_id}")

    def get_artifact_content_response(
        self,
        artifact_id: str,
        *,
        disposition: str = "inline",
        follow_redirects: bool = True,
    ) -> httpx.Response:
        response = self._client.get(
            f"/smr/artifacts/{artifact_id}/content",
            params={"disposition": disposition},
            follow_redirects=follow_redirects,
        )
        if response.status_code >= 400:
            snippet = response.text[:500] if response.text else ""
            raise SmrApiError(
                f"GET /smr/artifacts/{artifact_id}/content failed "
                f"({response.status_code}): {snippet}"
            )
        return response

    def get_artifact_content_bytes(
        self,
        artifact_id: str,
        *,
        disposition: str = "inline",
        follow_redirects: bool = True,
    ) -> bytes:
        response = self.get_artifact_content_response(
            artifact_id,
            disposition=disposition,
            follow_redirects=follow_redirects,
        )
        return response.content

    # Usage and ops -----------------------------------------------------

    def get_usage(self, project_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/projects/{project_id}/usage")

    def get_ops_status(
        self, project_id: str, *, include_done_tasks: bool | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if include_done_tasks is not None:
            params["include_done_tasks"] = int(include_done_tasks)
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/ops_status",
            params=params or None,
        )

    def get_run_logs(
        self,
        project_id: str,
        run_id: str,
        *,
        task_key: str | None = None,
        component: str | None = None,
        limit: int = 200,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Query VictoriaLogs for a specific run (structured, project-scoped).

        Returns ``{"ok", "query", "count", "records"}``.
        """
        params: dict[str, Any] = {"limit": limit}
        if task_key:
            params["task_key"] = task_key
        if component:
            params["component"] = component
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/{run_id}/logs",
            params=params,
        )

    def get_run_results(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Get run result summary: outcome, artifacts by type, debug log hint.

        Returns ``{"run_id", "state", "outcome", "artifacts_by_type",
        "latest_summary_artifact_id", "log_query_hint", ...}``.
        """
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/{run_id}/results",
        )

    def get_run_orchestrator_status(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Get orchestrator status for a run: turns, current phase, heartbeat.

        Returns::

            {
                "run_id", "run_state", "orchestrator_status",
                "claimed_by", "last_heartbeat_at",
                "current_phase", "elapsed_seconds", "last_tick_at", "agent",
                "turn_count",
                "current_turn": {"turn_number", "phase", "started_at", "completed", ...} | None,
                "turns": [{"turn_number", "phase", "started_at", "finished_at",
                            "completed", "error", "duration_seconds"}, ...],
                "log_query_hint",
            }
        """
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/{run_id}/orchestrator",
        )

    def get_project_git_status(self, project_id: str) -> dict[str, Any]:
        """Get read-only workspace git status for a project.

        Returns::

            {
                "project_id",
                "configured": bool,
                "commit_sha": str | None,
                "last_pushed_at": str | None,  # ISO-8601
                "default_branch": str | None,
                "vcs_provider": str | None,
                "remote_repo": str | None,
            }

        Storage internals (bucket, archive key) are not included.
        """
        return self._request_json("GET", f"/smr/projects/{project_id}/workspace/git")

    def search_victoria_logs(
        self,
        project_id: str,
        *,
        q: str | None = None,
        limit: int = 200,
        run_id: str | None = None,
        service: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if q:
            params["q"] = q
        if run_id:
            params["run_id"] = run_id
        if service:
            params["service"] = service
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/victoria-logs/search",
            params=params,
        )

    def list_run_log_archives(self, project_id: str, run_id: str) -> list[dict[str, Any]]:
        """List archived run log bundles (admin-scoped backend policy)."""
        data = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/{run_id}/logs/archives",
        )
        return _coerce_list(data, label="list_run_log_archives")

    def list_run_log_archives_typed(self, project_id: str, run_id: str) -> list[SmrRunLogArchive]:
        return [
            SmrRunLogArchive.from_dict(item)
            for item in self.list_run_log_archives(project_id, run_id)
        ]

    def get_run_economics(self, run_id: str) -> dict[str, Any]:
        """Fetch run economics details (admin-scoped backend policy)."""
        payload = self._request_json(
            "GET",
            f"/smr/admin/runs/{run_id}/economics",
        )
        return _coerce_dict(payload, label="get_run_economics")

    def get_run_economics_typed(self, run_id: str) -> SmrRunEconomics:
        return SmrRunEconomics.from_dict(self.get_run_economics(run_id))


def first_id(items: Iterable[dict[str, Any]], key: str) -> str | None:
    """Return the first non-empty string value for ``key`` across a dict iterable."""
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


ManagedResearchClient = SmrControlClient
