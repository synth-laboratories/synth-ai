"""Environment Pools — run coding agents in managed sandboxes.

**Status:** Alpha

This module provides the ``EnvironmentPoolsClient`` for creating pools,
launching rollouts, streaming SSE events, and capturing artifacts and usage.
"""

from __future__ import annotations

import json
import os
import warnings
from enum import Enum
from typing import Any, Iterator

from synth_ai.core.errors import PlanGatingError
from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import (
    BACKEND_URL_BASE,
    backend_me_url,
    join_url,
    normalize_backend_base,
)

__all__ = [
    # Schemas
    "AgentSpec",
    "ArtifactManifest",
    "ArtifactManifestEntry",
    "DatasetProvenance",
    "EnvironmentSpec",
    "EventEnvelope",
    "PoolConfigV1",
    "PoolDataSource",
    "PoolResources",
    "PoolResponse",
    "PoolTaskInstance",
    "PoolTemplate",
    "ReproMetadata",
    "RolloutHandle",
    "RolloutRequestV1",
    "SynthAPIError",
    "TaskRef",
    "TimeoutSpec",
    "UsageSnapshot",
    # Functions
    "create_rollout",
    "validate_rollout",
    "list_rollouts",
    "create_rollouts_batch",
    "get_rollout",
    "get_rollout_summary",
    "get_rollout_usage",
    "get_rollout_support_bundle",
    "replay_rollout",
    "stream_rollout_events",
    "stream_rollout_events_resilient",  # Alias with auto_reconnect=True
    "list_rollout_artifacts",
    "download_artifacts_zip",
    "fetch_artifact",
    "cancel_rollout",
    "list_pools",
    "get_pool",
    "create_pool",
    "update_pool",
    "delete_pool",
    "get_pool_metrics",
    "list_pool_tasks",
    "create_pool_task",
    "update_pool_task",
    "delete_pool_task",
    "get_queue_status",
    "get_capabilities",
    "get_openapi_schema",
    "get_schema_json",
    "store_credential",
    "list_credentials",
    "delete_credential",
    "rotate_credential",
    # Client + PoolTask
    "EnvironmentPoolsClient",
    "PoolTask",
    # Plan gating
    "PlanGatingError",
]

_API_PREFIX = "/api/v1/environment-pools"
_CREDENTIALS_PREFIX = "/api/v1/credentials"
_PUBLIC_API_PREFIX = "/v1"


class PoolTemplate(str, Enum):
    """Available pool templates for hosted containers."""

    HARBOR_CODING = "HARBOR_CODING"
    HARBOR_BROWSER = "HARBOR_BROWSER"
    ARCHIPELAGO = "ARCHIPELAGO"
    OPENENV = "OPENENV"


_TEMPLATE_POOL_TYPES: dict[str, str] = {
    PoolTemplate.HARBOR_CODING.value: "sandbox",
    PoolTemplate.HARBOR_BROWSER.value: "browser",
    PoolTemplate.ARCHIPELAGO.value: "archipelago",
    PoolTemplate.OPENENV.value: "openenv",
}

_TEMPLATE_BACKENDS: dict[str, str] = {
    PoolTemplate.HARBOR_CODING.value: "harbor",
    PoolTemplate.HARBOR_BROWSER.value: "kernel",
    PoolTemplate.ARCHIPELAGO.value: "archipelago",
    PoolTemplate.OPENENV.value: "openenv",
}


def _resolve_base_url(base_url: str | None, *, default: str) -> str:
    if base_url and base_url.strip():
        return normalize_backend_base(base_url)
    return normalize_backend_base(default)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key
    try:
        resolved = get_api_key("SYNTH_API_KEY", required=True)
    except Exception:
        resolved = os.environ.get("SYNTH_API_KEY", "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide or set SYNTH_API_KEY)")
    return resolved


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _request_headers(api_key: str, idempotency_key: str | None = None) -> dict[str, str]:
    headers = _auth_headers(api_key)
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _resolve_api_version(api_version: str | None, base_url: str) -> str:
    if api_version:
        return api_version
    env_version = os.environ.get("ENV_POOLS_API_VERSION", "").strip().lower()
    if env_version:
        return env_version
    return "v1"


def _url(base: str, path: str, *, api_version: str | None = None) -> str:
    version = _resolve_api_version(api_version, base)
    if version == "v1":
        return join_url(base, f"{_PUBLIC_API_PREFIX}/{path.lstrip('/')}")
    return join_url(base, f"{_API_PREFIX}/{path.lstrip('/')}")


def _cred_url(base: str, path: str = "") -> str:
    suffix = f"{_CREDENTIALS_PREFIX}/{path.lstrip('/')}" if path else _CREDENTIALS_PREFIX
    return join_url(base, suffix)


_PLAN_GATED_TIERS: set[str] = {"pro", "team", "enterprise"}
_PLAN_GATED_ALLOW_DEMO: bool = True

_UPGRADE_URL = "https://usesynth.ai/pricing"
_UPGRADE_MESSAGE = (
    "Environment Pools is available on Pro and Team plans. Upgrade at {url} to access this feature."
)


def _check_plan_access(
    *,
    api_key: str,
    backend_base: str,
    feature: str = "environment_pools",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Verify the caller's plan allows access to Environment Pools.

    Calls ``/api/v1/me`` and checks the plan tier.  Raises
    :class:`PlanGatingError` for Free/Trial plans and returns the
    account info dict on success.
    """
    import httpx

    url = backend_me_url(backend_base)
    try:
        resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except Exception:
        # If the /me endpoint is unreachable we fall through and let the
        # actual API call handle auth.  This keeps the SDK functional when
        # the /me endpoint is unavailable or the backend version is older.
        return {}

    plan = str(data.get("plan", data.get("tier", "free"))).lower()
    if plan in _PLAN_GATED_TIERS:
        return data
    if _PLAN_GATED_ALLOW_DEMO and plan == "demo":
        return data

    # Check feature flag overrides — allows free-tier orgs flagged in to bypass plan gating.
    # The /me response may include a "feature_flags" dict or list with per-feature overrides.
    feature_flags = data.get("feature_flags", {})
    if isinstance(feature_flags, dict):
        flag = feature_flags.get(feature)
        if isinstance(flag, bool) and flag:
            return data
        if isinstance(flag, dict) and flag.get("enabled", False):
            return data
    elif isinstance(feature_flags, list):
        for flag in feature_flags:
            if (
                isinstance(flag, dict)
                and flag.get("feature") == feature
                and flag.get("enabled", False)
            ):
                return data
            if isinstance(flag, str) and flag == feature:
                return data

    raise PlanGatingError(
        feature=feature,
        current_plan=plan,
        required_plans=("pro", "team"),
        upgrade_url=_UPGRADE_URL,
    )


def _raise_for_status_with_plan_check(response: Any) -> None:
    """Like ``_raise_for_status`` but converts HTTP 403 into :class:`PlanGatingError`."""
    status_code = getattr(response, "status_code", None)
    if status_code == 403:
        try:
            data = response.json()
        except Exception:
            data = {}
        error = data.get("error", data)
        code = error.get("code", "")
        # Only convert plan/feature gating 403s, not generic auth 403s
        if code in ("plan_required", "feature_not_available", "upgrade_required") or (
            "plan" in str(error.get("message", "")).lower()
            or "upgrade" in str(error.get("message", "")).lower()
        ):
            plan = error.get("current_plan", error.get("tier", "free"))
            raise PlanGatingError(
                feature="environment_pools",
                current_plan=str(plan),
                required_plans=("pro", "team"),
                upgrade_url=error.get("upgrade_url", _UPGRADE_URL),
            )
    _raise_for_status(response)


def _capture_sdk_error(exc: Exception, *, response: Any | None = None) -> None:
    try:
        import sentry_sdk  # type: ignore[import-not-found]
    except Exception:
        return
    if sentry_sdk.Hub.current.client is None:
        return
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("product", "environment_pools")
        scope.set_tag("backend_service", "sdk")
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                scope.set_tag("status_code", str(status_code))
            request_id = (
                response.headers.get("x-request-id") if hasattr(response, "headers") else None
            )
            if request_id:
                scope.set_tag("request_id", request_id)
            url = getattr(response, "url", None)
            if url:
                scope.set_tag("endpoint", str(url))
        sentry_sdk.capture_exception(exc)


def _raise_for_status(response: Any) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:
        _capture_sdk_error(exc, response=response)
        raise


# --- Schemas (v1) ---


try:
    from pydantic import BaseModel, Field, model_validator
except Exception:  # pragma: no cover - optional for minimal runtime
    BaseModel = object  # type: ignore[assignment]

    def Field(*args, **kwargs):  # type: ignore[no-redef] # noqa: N802
        return None

    def model_validator(*_args, **_kwargs):  # type: ignore[no-redef]
        def _wrap(fn):
            return fn

        return _wrap


class TaskRef(BaseModel):
    dataset: str = Field(..., description="Dataset identifier")
    task_id: str = Field(..., description="Task identifier within the dataset")
    version: str | None = Field(default=None, description="Dataset/task version")


_AGENT_MODEL_ALLOWLIST: dict[str, set[str]] = {
    "claude-code": {"claude-sonnet-4.5"},
    "opencode": {"claude-sonnet-4.5"},
    "codex": {"gpt-5.2-codex", "gpt-5.1-codex-mini", "gpt-5.3-codex-spark"},
}

_AGENT_DEFAULT_MODELS: dict[str, str] = {
    "claude-code": "claude-sonnet-4.5",
    "opencode": "claude-sonnet-4.5",
    "codex": "gpt-5.2-codex",
}


class AgentSpec(BaseModel):
    harness: str = Field(..., description="Agent harness (claude-code, opencode, codex, etc.)")
    harness_version: str | None = Field(default=None, description="Pinned harness version")
    model_id: str | None = Field(default=None, description="Provider model id")
    permission_mode: str | None = None
    agent_mode: str | None = None
    variant: str | None = None

    @classmethod
    def _validate_model(cls, harness: str, model_id: str | None) -> str:
        """Validate model_id against allowlist, returning resolved model."""
        default = _AGENT_DEFAULT_MODELS.get(harness)
        if model_id is None:
            if default is None:
                raise ValueError(f"Unknown harness {harness!r}; no default model available")
            return default
        allowed = _AGENT_MODEL_ALLOWLIST.get(harness)
        if allowed is not None and model_id not in allowed:
            raise ValueError(
                f"Model {model_id!r} is not allowed for harness {harness!r}. "
                f"Allowed models: {sorted(allowed)}"
            )
        return model_id

    @classmethod
    def preset(
        cls, name: str, *, model_id: str | None = None, harness_version: str | None = None
    ) -> AgentSpec:
        """Create an AgentSpec from a known harness preset name."""
        if name not in _AGENT_DEFAULT_MODELS:
            raise ValueError(
                f"Unknown preset {name!r}. Available presets: {sorted(_AGENT_DEFAULT_MODELS)}"
            )
        resolved_model = cls._validate_model(name, model_id)
        return cls(harness=name, model_id=resolved_model, harness_version=harness_version)

    @classmethod
    def claude_code(
        cls, *, model_id: str | None = None, harness_version: str | None = None
    ) -> AgentSpec:
        """Create an AgentSpec for the claude-code harness."""
        return cls.preset("claude-code", model_id=model_id, harness_version=harness_version)

    @classmethod
    def opencode(
        cls, *, model_id: str | None = None, harness_version: str | None = None
    ) -> AgentSpec:
        """Create an AgentSpec for the opencode harness."""
        return cls.preset("opencode", model_id=model_id, harness_version=harness_version)

    @classmethod
    def codex(cls, *, model_id: str | None = None, harness_version: str | None = None) -> AgentSpec:
        """Create an AgentSpec for the codex harness."""
        return cls.preset("codex", model_id=model_id, harness_version=harness_version)


class EnvironmentSpec(BaseModel):
    backend: str = Field(
        ..., description="Backend provider (harbor, browser, openenv, archipelago)"
    )
    docker_image: str | None = None


class TimeoutSpec(BaseModel):
    agent_sec: int | None = None
    verifier_sec: int | None = None
    build_sec: int | None = None


class PoolResources(BaseModel):
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None


class DatasetProvenance(BaseModel):
    git_url: str | None = None
    git_commit_id: str | None = None
    task_path: str | None = None
    dataset_ref: str | None = None
    task_id: str | None = None
    version: str | None = None


class ReproMetadata(BaseModel):
    git_ref: str | None = None
    git_commit_id: str | None = None
    image_digest: str | None = None
    agent_version: str | None = None
    sandbox_agent_version: str | None = None
    model_id: str | None = None
    dataset: DatasetProvenance | None = None


class PoolDataSource(BaseModel):
    type: str | None = None
    url: str | None = None
    git_ref: str | None = None
    subdir: str | None = None
    credential: str | None = None
    repo: str | None = None
    revision: str | None = None
    path: str | None = None
    upload_id: str | None = None


class PoolTaskInstance(BaseModel):
    task_id: str
    backend: str
    data_source: PoolDataSource | None = None
    docker_image: str | None = None
    registry_credential: str | None = None
    task_path: str | None = None
    env_vars: dict[str, str] | None = None
    resources: PoolResources | None = None
    harbor: dict[str, Any] | None = None
    harbor_input: dict[str, Any] | None = None
    browser: dict[str, Any] | None = None
    openenv_deployment: dict[str, Any] | None = None
    openenv_rollout: dict[str, Any] | None = None
    archipelago: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class RolloutRequestV1(BaseModel):
    task_ref: TaskRef
    agent: AgentSpec
    environment: EnvironmentSpec | None = None
    timeouts: TimeoutSpec | None = None
    pool_id: str | None = None
    pool_tags: list[str] | None = None
    priority: int | None = None
    dry_run: bool | None = None
    container_url: str | None = None
    container_id: str | None = None
    metadata: dict[str, Any] | None = None
    repro_metadata: ReproMetadata | None = None

    @model_validator(mode="after")
    def _validate_required(self) -> RolloutRequestV1:
        if not (self.pool_id or self.pool_tags or self.environment):
            raise ValueError("Provide pool_id, pool_tags, or environment")
        if self.container_url is None and self.container_id is None:
            raise ValueError("Provide either container_url or container_id")
        if self.timeouts is None:
            raise ValueError("timeouts is required")
        return self


class PoolConfigV1(BaseModel):
    pool_id: str
    pool_type: str
    org_id: str | None = None
    backend: str | None = None
    capacity: int | None = None
    concurrency: int | None = None
    priority_weight: int | None = None
    resources: PoolResources | None = None
    policy_tags: list[str] | None = None
    allowed_agents: list[str] | None = None
    allowed_models: list[str] | None = None
    tasks: list[PoolTaskInstance] | None = None
    max_queue: int | None = None
    max_running: int | None = None
    spend_limit_usd: float | None = None
    idle_timeout_sec: int | None = None
    max_session_age_sec: int | None = None
    reuse_policy: str | None = None
    prewarm: int | None = None


class EventEnvelope(BaseModel):
    """Canonical SSE event envelope."""

    id: str | None = Field(default=None, description="Event ID for resume")
    event: str = Field(
        default="message", description="Event type (lifecycle, agent, verifier, reward, heartbeat)"
    )
    seq: int | None = Field(default=None, description="Monotonic sequence number")
    timestamp: str | None = Field(default=None, description="ISO8601 timestamp")
    data: dict[str, Any] | str | None = Field(default=None, description="Event payload")


class ArtifactManifestEntry(BaseModel):
    """Single artifact in the manifest."""

    path: str = Field(..., description="Artifact path relative to rollout root")
    size_bytes: int | None = Field(default=None, description="File size in bytes")
    sha256: str | None = Field(default=None, description="SHA256 hash of contents")
    content_type: str | None = Field(default=None, description="MIME type")
    created_at: str | None = Field(default=None, description="ISO8601 creation timestamp")
    provenance: str | None = Field(
        default=None, description="Producer phase (agent, verifier, worker)"
    )
    download_url: str | None = Field(default=None, description="Stable download URL")


class ArtifactManifest(BaseModel):
    """Rollout artifact manifest."""

    rollout_id: str
    artifacts: list[ArtifactManifestEntry] = Field(default_factory=list)
    total_size_bytes: int | None = None
    created_at: str | None = None


class UsageSnapshot(BaseModel):
    """Usage/cost snapshot for a rollout."""

    rollout_id: str
    wall_time_sec: float | None = None
    sandbox_minutes: float | None = None
    browser_minutes: float | None = None
    bytes_uploaded: int | None = None
    bytes_downloaded: int | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    cost_usd: float | None = Field(default=None, description="Estimated cost in USD")
    provider_costs: dict[str, float] | None = Field(
        default=None, description="Per-provider cost breakdown"
    )


class PoolResponse(BaseModel):
    """Response from pool creation/retrieval with all pool metadata."""

    pool_id: str = Field(..., description="Unique pool identifier")
    pool_type: str | None = Field(default=None, description="Pool type (sandbox, browser, etc.)")
    backend: str | None = Field(default=None, description="Backend provider")
    template: str | None = Field(
        default=None, description="Pool template if using hosted containers"
    )
    container_url: str | None = Field(
        default=None, description="URL to use as container_url in rollouts"
    )
    container_id: str | None = Field(default=None, description="Associated container ID")
    org_id: str | None = Field(default=None, description="Organization ID")
    capacity: int | None = Field(default=None, description="Pool capacity")
    concurrency: int | None = Field(default=None, description="Concurrency limit")
    tasks: list[dict[str, Any]] | None = Field(default=None, description="Task definitions")
    status: str | None = Field(default=None, description="Pool status")
    created_at: str | None = Field(default=None, description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 last update timestamp")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoolResponse:
        """Create a PoolResponse from a raw API dict, extracting known fields."""
        return cls(
            pool_id=data.get("pool_id", data.get("id", "")),
            pool_type=data.get("pool_type"),
            backend=data.get("backend"),
            template=data.get("template"),
            container_url=data.get("container_url"),
            container_id=data.get("container_id"),
            org_id=data.get("org_id"),
            capacity=data.get("capacity"),
            concurrency=data.get("concurrency"),
            tasks=data.get("tasks"),
            status=data.get("status"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# --- Errors ---


class SynthAPIError(Exception):
    """Canonical error from Synth API."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        rollout_id: str | None = None,
        retryable: bool = False,
        retry_after_ms: int | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []
        self.request_id = request_id
        self.rollout_id = rollout_id
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.status_code = status_code

    @classmethod
    def from_response(cls, response: Any) -> SynthAPIError:
        """Parse error from httpx response."""
        status_code = getattr(response, "status_code", None)
        try:
            data = response.json()
        except Exception:
            data = {}
        error = data.get("error", data)
        return cls(
            code=error.get("code", "unknown_error"),
            message=error.get(
                "message", str(response.text if hasattr(response, "text") else response)
            ),
            details=error.get("details"),
            request_id=error.get("request_id") or response.headers.get("x-request-id"),
            rollout_id=error.get("rollout_id"),
            retryable=error.get("retryable", False),
            retry_after_ms=error.get("retry_after_ms"),
            status_code=status_code,
        )

    def __repr__(self) -> str:
        return f"SynthAPIError(code={self.code!r}, message={self.message!r}, retryable={self.retryable})"


# --- Rollouts ---


def _payload_from_request(request: Any) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        payload = request.model_dump(exclude_none=True, by_alias=True)  # type: ignore[no-any-return]
    elif hasattr(request, "dict"):
        payload = request.dict(exclude_none=True)  # type: ignore[no-any-return]
    elif isinstance(request, dict):
        payload = request
    else:
        raise TypeError("request must be a dict or a pydantic model")
    agent = payload.get("agent") if isinstance(payload, dict) else None
    if isinstance(agent, dict) and "kind" not in agent and "harness" in agent:
        updated = dict(agent)
        updated["kind"] = agent["harness"]
        payload["agent"] = updated
    return payload


def _rollout_id_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("rollout_id") or payload.get("trial_id")
    if isinstance(value, str) and value.strip():
        return value
    return None


def create_rollout(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any],
    idempotency_key: str | None = None,
    dry_run: bool | None = None,
    timeout: float = 120.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Create a new rollout."""
    warnings.warn(
        "create_rollout is intended for internal execution flows. "
        "Public SDK usage should be observability-only (get_rollout/stream_rollout_events).",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "rollouts", api_version=api_version)
    payload = _payload_from_request(request)
    if dry_run is not None:
        payload["dry_run"] = dry_run
    resp = httpx.post(
        url,
        headers=_request_headers(api_key, idempotency_key),
        json=payload,
        timeout=timeout,
    )
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def validate_rollout(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any],
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Validate a rollout request without executing it."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "rollouts:validate", api_version=api_version)
    payload = _payload_from_request(request)
    resp = httpx.post(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def list_rollouts(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    status: str | None = None,
    pool_id: str | None = None,
    dataset: str | None = None,
    task_id: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """List rollouts with optional filters."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "rollouts", api_version=api_version)
    params: dict[str, str] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = str(limit)
    if status is not None:
        params["status"] = status
    if pool_id is not None:
        params["pool_id"] = pool_id
    if dataset is not None:
        params["dataset"] = dataset
    if task_id is not None:
        params["task_id"] = task_id
    if created_after is not None:
        params["created_after"] = created_after
    if created_before is not None:
        params["created_before"] = created_before
    resp = httpx.get(url, headers=_auth_headers(api_key), params=params, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def create_rollouts_batch(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    requests: list[dict[str, Any]] | list[RolloutRequestV1],
    metadata: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    timeout: float = 120.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Create a batch of rollouts."""
    warnings.warn(
        "create_rollouts_batch is intended for internal execution flows. "
        "Public SDK usage should be observability-only (get_rollout/stream_rollout_events).",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "rollouts/batch", api_version=api_version)
    payload = {
        "requests": [_payload_from_request(req) for req in requests],
    }
    if metadata is not None:
        payload["metadata"] = metadata
    resp = httpx.post(
        url,
        headers=_request_headers(api_key, idempotency_key),
        json=payload,
        timeout=timeout,
    )
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_rollout(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Get rollout status."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_rollout_summary(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Get a rollout summary (latest status + last event)."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/summary", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_rollout_usage(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Get usage/cost snapshot for a rollout."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/usage", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def download_artifacts_zip(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    prefix: str | None = None,
    limit: int | None = None,
    timeout: float = 120.0,
    api_version: str | None = None,
) -> bytes:
    """Download a zip archive of rollout artifacts."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/artifacts.zip", api_version=api_version)
    params: dict[str, str] = {}
    if prefix is not None:
        params["prefix"] = prefix
    if limit is not None:
        params["limit"] = str(limit)
    resp = httpx.get(url, headers=_auth_headers(api_key), params=params, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    return resp.content


def get_rollout_support_bundle(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Fetch support bundle (config + routing + events + artifacts) for a rollout."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/support_bundle", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def replay_rollout(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    overrides: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float = 120.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Replay a rollout with optional overrides."""
    warnings.warn(
        "replay_rollout is intended for internal execution flows. "
        "Public SDK usage should be observability-only (get_rollout/stream_rollout_events).",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/replay", api_version=api_version)
    payload: dict[str, Any] = {}
    if overrides is not None:
        payload["overrides"] = overrides
    if metadata is not None:
        payload["metadata"] = metadata
    resp = httpx.post(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def stream_rollout_events(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    since: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    timeout: float | None = None,
    api_version: str | None = None,
    auto_reconnect: bool = False,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Stream rollout events via SSE. Yields parsed event dicts.

    Args:
        rollout_id: The rollout to stream events for.
        backend_base: Override backend URL.
        api_key: API key (or uses SYNTH_API_KEY env var).
        since: Start from events after this timestamp.
        cursor: Resume from this cursor (Last-Event-ID).
        limit: Maximum number of events to return.
        timeout: Read timeout in seconds.
        api_version: API version to use.
        auto_reconnect: If True, automatically reconnect on transient failures.
        max_retries: Maximum reconnection attempts (only if auto_reconnect=True).
        backoff_base: Base delay for exponential backoff in seconds.
        backoff_max: Maximum backoff delay in seconds.

    Yields:
        Event dicts with 'id', 'event', and 'data' fields.
    """
    import time

    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/events", api_version=api_version)

    last_event_id = cursor
    retries = 0

    while True:
        params: dict[str, str] = {}
        if since is not None:
            params["since"] = since
        if last_event_id is not None:
            params["cursor"] = last_event_id
        if limit is not None:
            params["limit"] = str(limit)

        headers = _auth_headers(api_key)
        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id

        try:
            with httpx.stream(
                "GET",
                url,
                headers=headers,
                params=params,
                timeout=httpx.Timeout(30.0, read=timeout),
            ) as response:
                _raise_for_status_with_plan_check(response)
                retries = 0  # Reset on successful connection
                event_data = ""
                event_id = ""
                event_type = ""
                for line in response.iter_lines():
                    if line.startswith("id:"):
                        event_id = line[3:].strip()
                    elif line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data = line[5:].strip()
                    elif line == "":
                        if event_data:
                            try:
                                parsed = json.loads(event_data)
                            except json.JSONDecodeError:
                                parsed = event_data
                            evt: dict[str, Any] = {"data": parsed}
                            if event_id:
                                evt["id"] = event_id
                                last_event_id = event_id  # Track for reconnect
                            if event_type:
                                evt["event"] = event_type
                            yield evt
                        event_data = ""
                        event_id = ""
                        event_type = ""
                # Stream ended normally
                return
        except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ConnectError) as e:
            _capture_sdk_error(e)
            if not auto_reconnect or retries >= max_retries:
                raise
            retries += 1
            delay = min(backoff_base * (2 ** (retries - 1)), backoff_max)
            time.sleep(delay)


def stream_rollout_events_resilient(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    since: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    timeout: float | None = None,
    api_version: str | None = None,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Stream rollout events with automatic reconnection on transient failures.

    This is a convenience wrapper around stream_rollout_events with auto_reconnect=True.
    """
    return stream_rollout_events(
        rollout_id,
        backend_base=backend_base,
        api_key=api_key,
        since=since,
        cursor=cursor,
        limit=limit,
        timeout=timeout,
        api_version=api_version,
        auto_reconnect=True,
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
    )


def list_rollout_artifacts(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    prefix: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """List artifacts for a rollout."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/artifacts", api_version=api_version)
    params: dict[str, str] = {}
    if prefix is not None:
        params["prefix"] = prefix
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = str(limit)
    resp = httpx.get(url, headers=_auth_headers(api_key), params=params, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def fetch_artifact(
    rollout_id: str,
    path: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    api_version: str | None = None,
) -> bytes:
    """Fetch a specific artifact by path."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/artifacts/{path.lstrip('/')}", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    return resp.content


def cancel_rollout(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Cancel a rollout (if supported by the backend)."""
    warnings.warn(
        "cancel_rollout is intended for internal execution flows. "
        "Public SDK usage should be observability-only (get_rollout/stream_rollout_events).",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/cancel", api_version=api_version)
    resp = httpx.post(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


class RolloutHandle:
    """Convenience wrapper for rollout observability."""

    def __init__(
        self,
        rollout_id: str,
        *,
        backend_base: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self.rollout_id = rollout_id
        self.backend_base = backend_base
        self.api_key = api_key
        self.api_version = api_version

    def get(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_rollout(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            timeout=timeout,
            api_version=self.api_version,
        )

    def summary(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_rollout_summary(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            timeout=timeout,
            api_version=self.api_version,
        )

    def wait(
        self,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        terminal: set[str] | None = None,
    ) -> dict[str, Any]:
        import time

        terminal_states = terminal or {"succeeded", "failed", "cancelled", "error", "completed"}
        deadline = time.time() + timeout
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.get()
            status = str(last.get("status", "")).lower()
            if status in terminal_states:
                return last
            time.sleep(poll_interval)
        return last

    def events(
        self,
        *,
        since: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        return stream_rollout_events(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            since=since,
            cursor=cursor,
            limit=limit,
            timeout=timeout,
            api_version=self.api_version,
        )

    def artifacts(
        self,
        *,
        prefix: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return list_rollout_artifacts(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            prefix=prefix,
            cursor=cursor,
            limit=limit,
            timeout=timeout,
            api_version=self.api_version,
        )

    def download_zip(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        timeout: float = 120.0,
    ) -> bytes:
        return download_artifacts_zip(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            prefix=prefix,
            limit=limit,
            timeout=timeout,
            api_version=self.api_version,
        )

    def download(self, path: str, *, timeout: float = 60.0) -> bytes:
        return fetch_artifact(
            self.rollout_id,
            path,
            backend_base=self.backend_base,
            api_key=self.api_key,
            timeout=timeout,
            api_version=self.api_version,
        )

    def support_bundle(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_rollout_support_bundle(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            timeout=timeout,
            api_version=self.api_version,
        )

    def replay(
        self,
        *,
        overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        return replay_rollout(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            overrides=overrides,
            metadata=metadata,
            timeout=timeout,
            api_version=self.api_version,
        )

    def cancel(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return cancel_rollout(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            timeout=timeout,
            api_version=self.api_version,
        )

    def usage(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_rollout_usage(
            self.rollout_id,
            backend_base=self.backend_base,
            api_key=self.api_key,
            timeout=timeout,
            api_version=self.api_version,
        )


# --- Pools ---


def list_pools(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    pool_type: str | None = None,
    tag: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> list[dict[str, Any]]:
    """List all pools."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "pools", api_version=api_version)
    params: dict[str, str] = {}
    if pool_type is not None:
        params["type"] = pool_type
    if tag is not None:
        params["tag"] = tag
    resp = httpx.get(url, headers=_auth_headers(api_key), params=params, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, list) else []


def get_pool(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Get pool details."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _normalize_tasks_for_template(tasks: list[Any], template: str | None) -> list[dict[str, Any]]:
    if not template:
        return [
            (_payload_from_request(task) if not isinstance(task, dict) else task) for task in tasks
        ]
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        payload = _payload_from_request(task)
        config = payload.get("config")
        if not isinstance(config, dict):
            config = {}
        config["template"] = template
        payload["config"] = config
        backend_value = _TEMPLATE_BACKENDS.get(template)
        if backend_value and "backend" not in payload:
            payload["backend"] = backend_value
        normalized.append(payload)
    return normalized


def create_pool(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any] | None = None,
    pool_id: str | None = None,
    template: PoolTemplate | str | None = None,
    tasks: list[dict[str, Any] | PoolTaskInstance] | None = None,
    pool_type: str | None = None,
    backend: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Create a new pool.

    You may pass a raw ``request`` dict (internal usage) or use the
    convenience parameters (``pool_id``, ``template``, ``tasks``). When
    ``template`` is provided, the helper fills ``pool_type``/``backend`` and
    injects the template into each task's config.
    """
    warnings.warn(
        "create_pool is intended for internal control-plane usage. "
        "Public SDK usage should rely on pool observability only.",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "pools", api_version=api_version)

    template_value: str | None = None
    if template is not None:
        template_value = template.value if isinstance(template, PoolTemplate) else str(template)

    if request is None:
        if pool_id is None:
            raise ValueError("pool_id is required when request is not provided")
        payload: dict[str, Any] = {"pool_id": pool_id}
        if template_value:
            payload["template"] = template_value
        if tasks is not None:
            payload["tasks"] = _normalize_tasks_for_template(tasks, template_value)
        if pool_type is not None:
            payload["pool_type"] = pool_type
        elif template_value and template_value in _TEMPLATE_POOL_TYPES:
            payload["pool_type"] = _TEMPLATE_POOL_TYPES[template_value]
        if backend is not None:
            payload["backend"] = backend
        elif template_value and template_value in _TEMPLATE_BACKENDS:
            payload["backend"] = _TEMPLATE_BACKENDS[template_value]
    else:
        payload = _payload_from_request(request)
        if template_value:
            payload.setdefault("template", template_value)
            if "pool_type" not in payload and template_value in _TEMPLATE_POOL_TYPES:
                payload["pool_type"] = _TEMPLATE_POOL_TYPES[template_value]
            if "backend" not in payload and template_value in _TEMPLATE_BACKENDS:
                payload["backend"] = _TEMPLATE_BACKENDS[template_value]
            if "tasks" in payload and isinstance(payload["tasks"], list):
                payload["tasks"] = _normalize_tasks_for_template(payload["tasks"], template_value)
    resp = httpx.post(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def update_pool(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any],
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Update pool configuration.

    The ``request`` dict may include any mutable pool fields such as
    ``capacity``, ``concurrency``, ``tasks``, etc.
    """
    warnings.warn(
        "update_pool is intended for internal control-plane usage. "
        "Public SDK usage should rely on pool observability only.",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}", api_version=api_version)
    payload = _payload_from_request(request)
    resp = httpx.put(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def delete_pool(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> None:
    """Delete a pool."""
    warnings.warn(
        "delete_pool is intended for internal control-plane usage. "
        "Public SDK usage should rely on pool observability only.",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}", api_version=api_version)
    resp = httpx.delete(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)


def get_pool_metrics(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Get pool metrics including queue depth and running count."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}/metrics", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def list_pool_tasks(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> list[dict[str, Any]]:
    """List task definitions for a pool."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}/tasks", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, list) else []


def create_pool_task(
    pool_id: str,
    *,
    request: dict[str, Any] | PoolTaskInstance,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Create a task definition in a pool."""
    warnings.warn(
        "create_pool_task is intended for internal control-plane usage. "
        "Public SDK usage should rely on pool observability only.",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}/tasks", api_version=api_version)
    payload = _payload_from_request(request)
    resp = httpx.post(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def update_pool_task(
    pool_id: str,
    task_id: str,
    *,
    request: dict[str, Any] | PoolTaskInstance,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Update a task definition in a pool."""
    warnings.warn(
        "update_pool_task is intended for internal control-plane usage. "
        "Public SDK usage should rely on pool observability only.",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}/tasks/{task_id}", api_version=api_version)
    payload = _payload_from_request(request)
    resp = httpx.put(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def delete_pool_task(
    pool_id: str,
    task_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> None:
    """Delete a task definition from a pool."""
    warnings.warn(
        "delete_pool_task is intended for internal control-plane usage. "
        "Public SDK usage should rely on pool observability only.",
        RuntimeWarning,
        stacklevel=2,
    )
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}/tasks/{task_id}", api_version=api_version)
    resp = httpx.delete(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)


# --- Queue ---


def get_queue_status(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Get queue status."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "queue/status", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_capabilities(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Discover supported pool types, agents, models, and features."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "capabilities", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_openapi_schema(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Fetch the OpenAPI schema published by the backend."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "openapi.json", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_schema_json(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    api_version: str | None = None,
) -> dict[str, Any]:
    """Fetch the JSON schema published by the backend."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "schema.json", api_version=api_version)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


# --- Credentials ---


def store_credential(
    *,
    credential_name: str,
    credential_type: str,
    credential_value: str | dict[str, Any],
    metadata: dict[str, Any] | None = None,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Store a new credential (DOCKER_REGISTRY, GITHUB_PAT, etc.).

    For DOCKER_REGISTRY credentials, ``credential_value`` should be a dict
    with ``username`` and ``token`` keys. It will be JSON-serialized
    automatically.
    """
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _cred_url(base)
    value = json.dumps(credential_value) if isinstance(credential_value, dict) else credential_value
    payload: dict[str, Any] = {
        "credential_name": credential_name,
        "credential_type": credential_type,
        "credential_value": value,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    resp = httpx.post(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def list_credentials(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """List credentials for the authenticated org (values are redacted)."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _cred_url(base)
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, list) else []


def delete_credential(
    credential_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> None:
    """Delete (deactivate) a credential."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _cred_url(base, credential_id)
    resp = httpx.delete(url, headers=_auth_headers(api_key), timeout=timeout)
    _raise_for_status_with_plan_check(resp)


def rotate_credential(
    credential_id: str,
    *,
    new_value: str | dict[str, Any],
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Rotate (update) a credential's value."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _cred_url(base, credential_id)
    value = json.dumps(new_value) if isinstance(new_value, dict) else new_value
    resp = httpx.put(
        url, headers=_auth_headers(api_key), json={"credential_value": value}, timeout=timeout
    )
    _raise_for_status_with_plan_check(resp)
    data = resp.json()
    return data if isinstance(data, dict) else {}


# --- PoolTask factory ---


class PoolTask:
    """Factory for creating PoolTaskInstance objects from common backends."""

    @staticmethod
    def from_docker(
        task_id: str,
        image: str,
        *,
        env: dict[str, str] | None = None,
        resources: PoolResources | dict[str, Any] | None = None,
        registry_credential: str | None = None,
        task_path: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> PoolTaskInstance:
        """Create a Docker/Harbor-backed task instance."""
        res = resources
        if isinstance(resources, dict):
            res = PoolResources(**resources)
        return PoolTaskInstance(
            task_id=task_id,
            backend="harbor",
            docker_image=image,
            env_vars=env,
            resources=res,  # type: ignore[arg-type]
            registry_credential=registry_credential,
            task_path=task_path,
            config=config,
        )

    @staticmethod
    def from_openenv(
        task_id: str,
        container_url: str,
        *,
        env: dict[str, str] | None = None,
        resources: PoolResources | dict[str, Any] | None = None,
        openenv_deployment: dict[str, Any] | None = None,
        openenv_rollout: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> PoolTaskInstance:
        """Create an OpenEnv-backed task instance."""
        res = resources
        if isinstance(resources, dict):
            res = PoolResources(**resources)
        return PoolTaskInstance(
            task_id=task_id,
            backend="openenv",
            env_vars=env,
            resources=res,  # type: ignore[arg-type]
            openenv_deployment=openenv_deployment or {"container_url": container_url},
            openenv_rollout=openenv_rollout,
            config=config,
        )

    @staticmethod
    def from_browser(
        task_id: str,
        container_url: str,
        *,
        profile: str | None = None,
        env: dict[str, str] | None = None,
        resources: PoolResources | dict[str, Any] | None = None,
        browser: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> PoolTaskInstance:
        """Create a browser-backed task instance."""
        res = resources
        if isinstance(resources, dict):
            res = PoolResources(**resources)
        browser_config = browser or {"container_url": container_url}
        if profile is not None:
            browser_config.setdefault("profile", profile)
        return PoolTaskInstance(
            task_id=task_id,
            backend="browser",
            env_vars=env,
            resources=res,  # type: ignore[arg-type]
            browser=browser_config,
            config=config,
        )


# --- EnvironmentPoolsClient ---


class _RolloutsNamespace:
    """Namespace for rollout operations on EnvironmentPoolsClient."""

    def __init__(
        self, *, backend_base: str | None, api_key: str | None, api_version: str | None
    ) -> None:
        self._backend_base = backend_base
        self._api_key = api_key
        self._api_version = api_version

    def create(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        dry_run: bool | None = None,
        timeout: float = 120.0,
    ) -> RolloutHandle:
        """Create a rollout and return a RolloutHandle."""
        data = create_rollout(
            backend_base=self._backend_base,
            api_key=self._api_key,
            request=request,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
            timeout=timeout,
            api_version=self._api_version,
        )
        rollout_id = _rollout_id_from_payload(data) or data.get("id", "")
        return RolloutHandle(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            api_version=self._api_version,
        )

    def get(self, rollout_id: str, *, timeout: float = 30.0) -> RolloutHandle:
        """Get a RolloutHandle for an existing rollout."""
        # Verify it exists by fetching it
        get_rollout(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )
        return RolloutHandle(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            api_version=self._api_version,
        )

    def list(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        status: str | None = None,
        pool_id: str | None = None,
        dataset: str | None = None,
        task_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """List rollouts with optional filters."""
        return list_rollouts(
            backend_base=self._backend_base,
            api_key=self._api_key,
            cursor=cursor,
            limit=limit,
            status=status,
            pool_id=pool_id,
            dataset=dataset,
            task_id=task_id,
            created_after=created_after,
            created_before=created_before,
            timeout=timeout,
            api_version=self._api_version,
        )

    def events(
        self,
        rollout_id: str,
        *,
        since: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
        auto_reconnect: bool = True,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        """Stream rollout events via SSE (auto-reconnect by default)."""
        return stream_rollout_events(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            since=since,
            cursor=cursor,
            limit=limit,
            timeout=timeout,
            api_version=self._api_version,
            auto_reconnect=auto_reconnect,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
        )

    def cancel(self, rollout_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Cancel a rollout."""
        return cancel_rollout(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def artifacts(
        self,
        rollout_id: str,
        *,
        prefix: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """List artifacts for a rollout."""
        return list_rollout_artifacts(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            prefix=prefix,
            cursor=cursor,
            limit=limit,
            timeout=timeout,
            api_version=self._api_version,
        )

    def usage(self, rollout_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Get usage/cost snapshot for a rollout."""
        return get_rollout_usage(
            rollout_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )


class _PoolTasksNamespace:
    """Sub-namespace for pool task operations."""

    def __init__(
        self, *, backend_base: str | None, api_key: str | None, api_version: str | None
    ) -> None:
        self._backend_base = backend_base
        self._api_key = api_key
        self._api_version = api_version

    def list(self, pool_id: str, *, timeout: float = 30.0) -> list[dict[str, Any]]:
        return list_pool_tasks(
            pool_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def create(
        self, pool_id: str, *, request: dict[str, Any] | PoolTaskInstance, timeout: float = 30.0
    ) -> dict[str, Any]:
        return create_pool_task(
            pool_id,
            request=request,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def update(
        self,
        pool_id: str,
        task_id: str,
        *,
        request: dict[str, Any] | PoolTaskInstance,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return update_pool_task(
            pool_id,
            task_id,
            request=request,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def delete(self, pool_id: str, task_id: str, *, timeout: float = 30.0) -> None:
        return delete_pool_task(
            pool_id,
            task_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )


class _PoolsNamespace:
    """Namespace for pool operations on EnvironmentPoolsClient."""

    def __init__(
        self, *, backend_base: str | None, api_key: str | None, api_version: str | None
    ) -> None:
        self._backend_base = backend_base
        self._api_key = api_key
        self._api_version = api_version
        self.tasks = _PoolTasksNamespace(
            backend_base=backend_base, api_key=api_key, api_version=api_version
        )

    def list(
        self, *, pool_type: str | None = None, tag: str | None = None, timeout: float = 30.0
    ) -> list[PoolResponse]:
        """List all pools, returning PoolResponse objects."""
        data = list_pools(
            backend_base=self._backend_base,
            api_key=self._api_key,
            pool_type=pool_type,
            tag=tag,
            timeout=timeout,
            api_version=self._api_version,
        )
        return [PoolResponse.from_dict(pool) for pool in data]

    def list_raw(
        self, *, pool_type: str | None = None, tag: str | None = None, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """List all pools, returning raw dicts."""
        return list_pools(
            backend_base=self._backend_base,
            api_key=self._api_key,
            pool_type=pool_type,
            tag=tag,
            timeout=timeout,
            api_version=self._api_version,
        )

    def get(self, pool_id: str, *, timeout: float = 30.0) -> PoolResponse:
        """Get pool details as a PoolResponse."""
        data = get_pool(
            pool_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )
        return PoolResponse.from_dict(data)

    def get_raw(self, pool_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Get pool details as a raw dict."""
        return get_pool(
            pool_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def create(
        self,
        *,
        request: dict[str, Any] | None = None,
        pool_id: str | None = None,
        template: PoolTemplate | str | None = None,
        tasks: list[dict[str, Any] | PoolTaskInstance] | None = None,
        pool_type: str | None = None,
        backend: str | None = None,
        timeout: float = 30.0,
    ) -> PoolResponse:
        """Create a new pool, returning a PoolResponse."""
        data = create_pool(
            backend_base=self._backend_base,
            api_key=self._api_key,
            request=request,
            pool_id=pool_id,
            template=template,
            tasks=tasks,
            pool_type=pool_type,
            backend=backend,
            timeout=timeout,
            api_version=self._api_version,
        )
        return PoolResponse.from_dict(data)

    def create_raw(
        self,
        *,
        request: dict[str, Any] | None = None,
        pool_id: str | None = None,
        template: PoolTemplate | str | None = None,
        tasks: list[dict[str, Any] | PoolTaskInstance] | None = None,
        pool_type: str | None = None,
        backend: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Create a new pool, returning a raw dict."""
        return create_pool(
            backend_base=self._backend_base,
            api_key=self._api_key,
            request=request,
            pool_id=pool_id,
            template=template,
            tasks=tasks,
            pool_type=pool_type,
            backend=backend,
            timeout=timeout,
            api_version=self._api_version,
        )

    def update(
        self, pool_id: str, *, request: dict[str, Any], timeout: float = 30.0
    ) -> PoolResponse:
        """Update pool configuration, returning a PoolResponse."""
        data = update_pool(
            pool_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            request=request,
            timeout=timeout,
            api_version=self._api_version,
        )
        return PoolResponse.from_dict(data)

    def update_raw(
        self, pool_id: str, *, request: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        """Update pool configuration, returning a raw dict."""
        return update_pool(
            pool_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            request=request,
            timeout=timeout,
            api_version=self._api_version,
        )

    def delete(self, pool_id: str, *, timeout: float = 30.0) -> None:
        return delete_pool(
            pool_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )


class _CredentialsNamespace:
    """Namespace for credential operations on EnvironmentPoolsClient."""

    def __init__(self, *, backend_base: str | None, api_key: str | None) -> None:
        self._backend_base = backend_base
        self._api_key = api_key

    def store(
        self,
        *,
        credential_name: str,
        credential_type: str,
        credential_value: str | dict[str, Any],
        metadata: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return store_credential(
            credential_name=credential_name,
            credential_type=credential_type,
            credential_value=credential_value,
            metadata=metadata,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
        )

    def list(self, *, timeout: float = 30.0) -> list[dict[str, Any]]:
        return list_credentials(
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
        )

    def delete(self, credential_id: str, *, timeout: float = 30.0) -> None:
        return delete_credential(
            credential_id,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
        )

    def rotate(
        self,
        credential_id: str,
        *,
        new_value: str | dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return rotate_credential(
            credential_id,
            new_value=new_value,
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
        )


class _CapabilitiesNamespace:
    """Namespace for capabilities/schema operations on EnvironmentPoolsClient."""

    def __init__(
        self, *, backend_base: str | None, api_key: str | None, api_version: str | None
    ) -> None:
        self._backend_base = backend_base
        self._api_key = api_key
        self._api_version = api_version

    def get(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_capabilities(
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def openapi_schema(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_openapi_schema(
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )

    def schema_json(self, *, timeout: float = 30.0) -> dict[str, Any]:
        return get_schema_json(
            backend_base=self._backend_base,
            api_key=self._api_key,
            timeout=timeout,
            api_version=self._api_version,
        )


class EnvironmentPoolsClient:
    """Consolidated client for the Environment Pools API.

    Provides sub-namespaces for rollouts, pools, credentials, and capabilities.

    Usage::

        client = EnvironmentPoolsClient(api_key="sk_...")
        handle = client.rollouts.get("rollout-123")
        pools = client.pools.list()
        caps = client.capabilities.get()
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        api_version: str | None = None,
        skip_plan_check: bool = False,
    ) -> None:
        self._api_key = api_key
        self._backend_base = backend_base
        self._api_version = api_version

        # Eagerly verify plan access unless explicitly skipped.
        if not skip_plan_check:
            resolved_key = _resolve_api_key(api_key)
            resolved_base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
            self._account_info = _check_plan_access(
                api_key=resolved_key,
                backend_base=resolved_base,
            )
        else:
            self._account_info: dict[str, Any] = {}

        self.rollouts = _RolloutsNamespace(
            backend_base=backend_base, api_key=api_key, api_version=api_version
        )
        self.pools = _PoolsNamespace(
            backend_base=backend_base, api_key=api_key, api_version=api_version
        )
        self.credentials = _CredentialsNamespace(backend_base=backend_base, api_key=api_key)
        self.capabilities = _CapabilitiesNamespace(
            backend_base=backend_base, api_key=api_key, api_version=api_version
        )
