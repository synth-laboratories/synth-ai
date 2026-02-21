"""Managed Research control-plane SDK client.

This module exposes a focused client for controlling Synth Managed Research (SMR)
projects and runs through public API routes.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.container.auth import encrypt_for_backend

ACTIVE_RUN_STATES = {"queued", "planning", "executing", "blocked", "finalizing", "running"}
DEFAULT_TIMEOUT_SECONDS = 30.0

__all__ = [
    "ACTIVE_RUN_STATES",
    "DEFAULT_TIMEOUT_SECONDS",
    "ManagedResearchClient",
    "SmrApiError",
    "SmrControlClient",
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


@dataclass
class SmrControlClient:
    """SMR control-plane client with compatibility fallbacks."""

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

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        data = self._request_json(
            "GET",
            "/smr/projects",
            params={"include_archived": int(include_archived)},
        )
        return _coerce_list(data, label="list_projects")

    def get_project(self, project_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/projects/{project_id}")

    def patch_project(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("PATCH", f"/smr/projects/{project_id}", json_body=payload)

    def get_project_status(self, project_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/projects/{project_id}/status")

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

        payload: dict[str, Any] = {
            "provider": provider,
            "funding_source": funding_source,
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
        if response.status_code == 422 and "encrypted_key_b64" in payload and provider_api_key:
            fallback_payload = {
                "provider": provider,
                "funding_source": funding_source,
                "api_key": provider_api_key,
            }
            response = self._client.post(
                f"/smr/projects/{project_id}/provider_keys",
                json=fallback_payload,
            )

        if response.status_code >= 400:
            snippet = response.text[:500] if response.text else ""
            raise SmrApiError(
                f"POST /smr/projects/{project_id}/provider_keys failed "
                f"({response.status_code}): {snippet}"
            )
        return response.json() if response.content else {}

    def provider_key_status(self, project_id: str, provider: str, funding_source: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"/smr/projects/{project_id}/provider_keys/{provider}/{funding_source}/status",
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

        payload: dict[str, Any] = {"files": payload_files}
        if isinstance(dataset_ref, str) and dataset_ref.strip():
            payload["dataset_ref"] = dataset_ref.strip()

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
        file_payloads: dict[str, tuple[bytes, str | None]] = {}
        for file in files:
            if not isinstance(file, dict):
                raise ValueError("each file entry must be a JSON object")
            path = str(file.get("path") or "").strip()
            if not path:
                raise ValueError("each file entry requires non-empty 'path'")

            content = file.get("content")
            if isinstance(content, str):
                content_bytes = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                content_bytes = bytes(content)
            else:
                raise ValueError(f"file '{path}' requires 'content' as str|bytes")

            entry: dict[str, str] = {"path": path}
            content_type = file.get("content_type")
            resolved_content_type: str | None = None
            if isinstance(content_type, str) and content_type.strip():
                resolved_content_type = content_type.strip()
                entry["content_type"] = resolved_content_type

            request_files.append(entry)
            file_payloads[path] = (content_bytes, resolved_content_type)

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
                    raise SmrApiError(f"starting-data upload response returned unknown path '{path}'")
                content_bytes, content_type = payload
                headers: dict[str, str] = {}
                if content_type:
                    headers["Content-Type"] = content_type
                response = upload_client.put(upload_url, content=content_bytes, headers=headers or None)
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
                    "content": file_path.read_bytes(),
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
        """
        payload: dict[str, Any] = {}
        if timebox_seconds is not None:
            payload["timebox_seconds"] = int(timebox_seconds)
        if agent_model and agent_model.strip():
            payload["agent_model"] = agent_model.strip()
        if agent_kind and agent_kind.strip():
            payload["agent_kind"] = agent_kind.strip().lower()
        return self._request_json("POST", f"/smr/projects/{project_id}/trigger", json_body=payload)

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

    def list_active_runs(self, project_id: str) -> list[dict[str, Any]]:
        runs = self.list_runs(project_id)
        out: list[dict[str, Any]] = []
        for run in runs:
            state = str(run.get("state") or "").strip().lower()
            if state in ACTIVE_RUN_STATES:
                out.append(run)
        return out

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

    def pause_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/runs/{run_id}/pause")

    def resume_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/runs/{run_id}/resume")

    def stop_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/smr/runs/{run_id}/stop")

    # Questions + approvals --------------------------------------------

    def list_project_questions(self, project_id: str, *, status_filter: str = "pending") -> list[dict[str, Any]]:
        data = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/questions",
            params={"status_filter": status_filter},
        )
        return _coerce_list(data, label="list_project_questions")

    def list_run_questions(self, run_id: str, *, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/questions",
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_questions")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/questions")
        return _coerce_list(canonical, label="canonical_list_run_questions")

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
            scoped_path = f"/smr/projects/{project_id}/runs/{run_id}/questions/{question_id}/respond"
            scoped = self._request_json("POST", scoped_path, json_body=payload, allow_not_found=True)
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/questions/{question_id}/respond",
            json_body=payload,
        )

    def list_project_approvals(self, project_id: str, *, status_filter: str = "pending") -> list[dict[str, Any]]:
        data = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/approvals",
            params={"status_filter": status_filter},
        )
        return _coerce_list(data, label="list_project_approvals")

    def list_run_approvals(self, run_id: str, *, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals",
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_approvals")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/approvals")
        return _coerce_list(canonical, label="canonical_list_run_approvals")

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
            scoped_path = f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/approve"
            scoped = self._request_json("POST", scoped_path, json_body=payload, allow_not_found=True)
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
            scoped = self._request_json("POST", scoped_path, json_body=payload, allow_not_found=True)
            if scoped is not None:
                return scoped
        return self._request_json(
            "POST",
            f"/smr/runs/{run_id}/approvals/{approval_id}/deny",
            json_body=payload,
        )

    # Artifacts ---------------------------------------------------------

    def list_run_artifacts(self, run_id: str, *, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/artifacts",
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_list(scoped, label="project_scoped_list_run_artifacts")
        canonical = self._request_json("GET", f"/smr/runs/{run_id}/artifacts")
        return _coerce_list(canonical, label="canonical_list_run_artifacts")

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/smr/artifacts/{artifact_id}")

    def get_artifact_content_response(
        self,
        artifact_id: str,
        *,
        disposition: str = "inline",
        follow_redirects: bool = False,
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
        follow_redirects: bool = False,
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

    def get_ops_status(self, project_id: str, *, include_done_tasks: bool | None = None) -> dict[str, Any]:
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


def first_id(items: Iterable[dict[str, Any]], key: str) -> str | None:
    """Return the first non-empty string value for ``key`` across a dict iterable."""
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


ManagedResearchClient = SmrControlClient
