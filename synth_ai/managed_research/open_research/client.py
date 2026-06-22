"""HTTP client for Open Research v1.

Public, narrow surface. One method per locked endpoint. Auth header
plumbing follows the contract:

- Signed-in caller — ``Authorization: Bearer <synth_api_key>``.
- Anonymous caller — ``X-OR-Fingerprint: <fingerprint>``; backend mirrors
  the value into ``submitter.fingerprint`` for ``submit_question``.

Read endpoints are public; auth headers are still attached when
available so the backend can scope ``get_submission`` to the submitter.
"""

from __future__ import annotations

import hashlib
import json as _json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from synth_ai.managed_research._internal.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.open_research.errors import (
    parse_open_research_error_envelope,
)
from synth_ai.managed_research.open_research.models import (
    BundleDownloadResult,
    ExperimentDetail,
    ExperimentStatusFilter,
    ListExperimentsResponse,
    ListProjectsResponse,
    ListQueuesResponse,
    ProjectDetail,
    ReceiptPayload,
    SubmissionDetail,
    SubmissionResponse,
    SubmitQuestionArgs,
)

OPEN_RESEARCH_BASE = "/api/open-research/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_BUNDLE_TIMEOUT_SECONDS = 600.0


def _resolve_backend_base(backend_base: str | None) -> str:
    candidate = str(backend_base or BACKEND_URL_BASE).strip()
    if not candidate:
        candidate = "https://api.usesynth.ai"
    return normalize_backend_base(candidate).rstrip("/")


def _headers(
    *,
    api_key: str | None,
    fingerprint: str | None,
    accept_json: bool = True,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if accept_json:
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if fingerprint:
        headers["X-OR-Fingerprint"] = fingerprint
    return headers


def _raise_typed(response: httpx.Response) -> None:
    """Map a non-2xx response onto OpenResearchError when possible."""
    try:
        payload = response.json()
    except (_json.JSONDecodeError, ValueError):
        payload = None
    typed = parse_open_research_error_envelope(
        payload,
        status_code=response.status_code,
        response_text=response.text,
    )
    if typed is not None:
        raise typed
    # Contract says all non-2xx responses carry the envelope. A response
    # that doesn't is a backend bug or a non-contract surface; surface
    # the raw status so the caller can debug.
    message = (
        f"Open Research {response.request.method} "
        f"{response.request.url.path} failed with {response.status_code}"
    )
    raise SmrApiError(
        message,
        status_code=response.status_code,
        response_text=response.text,
    )


def prompt_hash(prompt: str) -> str:
    """Normalized prompt hash used for client-side idempotency display.

    The backend recomputes its own idempotency key. We expose the
    normalization here so MCP callers can confirm the contract's
    deduplication semantics ("same prompt → same submission") without
    guessing.
    """
    normalized = " ".join(prompt.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class OpenResearchClient:
    """Thin Open Research v1 client.

    The ``api_key`` and ``fingerprint`` are both optional. Anonymous
    submissions to the 1h OED queue need a fingerprint; signed-in
    callers need an api key with the ``smr_open_ended_discovery``
    entitlement. Read endpoints accept either, or neither.
    """

    api_key: str | None = None
    fingerprint: str | None = None
    backend_base: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    bundle_timeout_seconds: float = DEFAULT_BUNDLE_TIMEOUT_SECONDS
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend_base = _resolve_backend_base(self.backend_base)
        self._client = httpx.Client(
            base_url=self.backend_base,
            headers=_headers(
                api_key=self.api_key,
                fingerprint=self.fingerprint,
                accept_json=True,
            ),
            timeout=self.timeout_seconds,
        )

    def __enter__(self) -> OpenResearchClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ---- internals ----------------------------------------------------

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                json=json_body,
            )
        except httpx.TimeoutException as exc:
            raise SmrApiError(f"{method} {path} timed out") from exc
        except httpx.TransportError as exc:
            raise SmrApiError(
                f"{method} {path} failed: network error ({type(exc).__name__})"
            ) from exc
        if response.is_error:
            _raise_typed(response)
        if not response.content:
            return {}
        try:
            return response.json()
        except (_json.JSONDecodeError, ValueError) as exc:
            raise SmrApiError(
                f"{method} {path} returned a non-JSON response",
                status_code=response.status_code,
                response_text=response.text,
            ) from exc

    # ---- endpoints ----------------------------------------------------

    def list_projects(self) -> ListProjectsResponse:
        payload = self._request_json("GET", f"{OPEN_RESEARCH_BASE}/projects")
        return ListProjectsResponse.model_validate(payload)

    def get_project(self, slug: str) -> ProjectDetail:
        normalized = (slug or "").strip()
        if not normalized:
            raise ValueError("'slug' is required")
        payload = self._request_json("GET", f"{OPEN_RESEARCH_BASE}/projects/{normalized}")
        return ProjectDetail.model_validate(payload)

    def list_queues(self, *, project_slug: str | None = None) -> ListQueuesResponse:
        params: dict[str, Any] = {}
        if project_slug and project_slug.strip():
            params["project_slug"] = project_slug.strip()
        payload = self._request_json(
            "GET",
            f"{OPEN_RESEARCH_BASE}/queues",
            params=params or None,
        )
        return ListQueuesResponse.model_validate(payload)

    def submit_question(self, args: SubmitQuestionArgs) -> SubmissionResponse:
        body: dict[str, Any] = {
            "project_slug": args.project_slug,
            "queue_id": args.queue_id,
            "prompt": args.prompt,
            "hypothesis": args.hypothesis,
            "metric_target": {
                "name": args.metric_target.name,
                "operator": args.metric_target.operator,
                "value": args.metric_target.value,
            },
            "deo_kind": args.deo_kind,
            "rubric_acknowledged": args.rubric_acknowledged,
            "submitter": {
                "handle": args.submitter_handle,
                "fingerprint": args.submitter_fingerprint or self.fingerprint,
            },
        }
        payload = self._request_json(
            "POST",
            f"{OPEN_RESEARCH_BASE}/submissions",
            json_body=body,
        )
        return SubmissionResponse.model_validate(payload)

    def get_submission(self, submission_id: str) -> SubmissionDetail:
        normalized = (submission_id or "").strip()
        if not normalized:
            raise ValueError("'submission_id' is required")
        payload = self._request_json("GET", f"{OPEN_RESEARCH_BASE}/submissions/{normalized}")
        return SubmissionDetail.model_validate(payload)

    def list_experiments(
        self,
        *,
        project_slug: str | None = None,
        status: ExperimentStatusFilter | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ListExperimentsResponse:
        params: dict[str, Any] = {}
        if project_slug and project_slug.strip():
            params["project_slug"] = project_slug.strip()
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor and cursor.strip():
            params["cursor"] = cursor.strip()
        payload = self._request_json(
            "GET",
            f"{OPEN_RESEARCH_BASE}/experiments",
            params=params or None,
        )
        return ListExperimentsResponse.model_validate(payload)

    def get_experiment(self, experiment_id: str) -> ExperimentDetail:
        normalized = (experiment_id or "").strip()
        if not normalized:
            raise ValueError("'experiment_id' is required")
        payload = self._request_json("GET", f"{OPEN_RESEARCH_BASE}/experiments/{normalized}")
        return ExperimentDetail.model_validate(payload)

    def get_receipt(self, experiment_id: str) -> ReceiptPayload:
        normalized = (experiment_id or "").strip()
        if not normalized:
            raise ValueError("'experiment_id' is required")
        payload = self._request_json(
            "GET", f"{OPEN_RESEARCH_BASE}/experiments/{normalized}/receipt"
        )
        return ReceiptPayload.model_validate(payload)

    def download_bundle(
        self,
        experiment_id: str,
        dest_path: str | Path,
        *,
        timeout_seconds: float | None = None,
    ) -> BundleDownloadResult:
        normalized = (experiment_id or "").strip()
        if not normalized:
            raise ValueError("'experiment_id' is required")
        path = Path(dest_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        request_path = f"{OPEN_RESEARCH_BASE}/experiments/{normalized}/bundle"
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.bundle_timeout_seconds
        )
        hasher = hashlib.sha256()
        bytes_written = 0
        content_type: str | None = None
        try:
            with (
                self._client.stream(
                    "GET",
                    request_path,
                    timeout=effective_timeout,
                ) as response,
                path.open("wb") as handle,
            ):
                if response.is_error:
                    response.read()
                    _raise_typed(response)
                content_type = response.headers.get("content-type")
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    hasher.update(chunk)
                    handle.write(chunk)
                    bytes_written += len(chunk)
        except httpx.TimeoutException as exc:
            raise SmrApiError(f"GET {request_path} timed out after {effective_timeout}s") from exc
        except httpx.TransportError as exc:
            raise SmrApiError(
                f"GET {request_path} failed: network error ({type(exc).__name__})"
            ) from exc
        return BundleDownloadResult(
            experiment_id=normalized,
            output_path=str(path),
            bytes_written=bytes_written,
            sha256=hasher.hexdigest(),
            content_type=content_type,
        )


__all__ = [
    "DEFAULT_BUNDLE_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "OPEN_RESEARCH_BASE",
    "OpenResearchClient",
    "prompt_hash",
]
