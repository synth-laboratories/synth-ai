"""Canonical policy optimization SDK clients.

This module exposes explicit offline-job and online-session clients that target
versioned routes:
- `/{v1|v2}/policy-optimization/systems`
- `/{v1|v2}/offline/jobs`
- `/{v1|v2}/online/sessions`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Sequence
from urllib.parse import urlencode

import httpx

from synth_ai.core.errors import HTTPError, UsageLimitError, ValidationError
from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.optimization_routes import (
    admin_failures_query_path,
    admin_optimizer_events_path,
    admin_victoria_logs_query_path,
    candidates_submit_path,
    failures_query_path,
    normalize_api_version,
    offline_job_path,
    offline_job_queue_default_plan_path,
    offline_job_queue_rollout_drain_path,
    offline_job_queue_rollout_limiter_status_path,
    offline_job_queue_rollout_metrics_path,
    offline_job_queue_rollout_policy_path,
    offline_job_queue_rollout_retry_path,
    offline_job_queue_rollouts_path,
    offline_job_queue_trial_path,
    offline_job_queue_trials_path,
    offline_job_queue_trials_reorder_path,
    offline_job_state_baseline_info_path,
    offline_job_state_envelope_path,
    offline_jobs_base,
    online_session_path,
    online_sessions_base,
    optimizer_events_path,
    policy_system_path,
    policy_systems_base,
    runtime_compatibility_path,
    runtime_container_rollout_checkpoint_dump_path,
    runtime_container_rollout_checkpoint_restore_path,
    runtime_queue_contract_path,
    runtime_queue_rollout_expire_leases_path,
    runtime_queue_rollout_lease_path,
    runtime_queue_rollout_path,
    runtime_queue_rollouts_path,
    runtime_queue_trial_path,
    runtime_queue_trials_path,
    runtime_session_queue_contract_path,
    runtime_session_queue_rollout_expire_leases_path,
    runtime_session_queue_rollout_lease_path,
    runtime_session_queue_rollout_path,
    runtime_session_queue_rollouts_path,
    runtime_session_queue_trial_path,
    runtime_session_queue_trials_path,
)
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync

ApiVersion = Literal["v1", "v2"]
PromptOptVersion = Literal["v1", "v2"]
PromptOptFallbackPolicy = Literal["none", "preflight_only", "init_only", "preflight_or_init"]


def _resolve_backend_url(backend_url: Optional[str]) -> str:
    return (backend_url or BACKEND_URL_BASE).rstrip("/")


def _resolve_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    env_key = os.getenv("SYNTH_API_KEY")
    if not env_key:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY env var)")
    return env_key


def _resolve_api_version(api_version: Optional[str] = None) -> ApiVersion:
    raw = (api_version or os.getenv("SYNTH_POLICY_API_VERSION") or "v1").strip().lower()
    if raw not in {"v1", "v2"}:
        raise ValueError("api_version must be 'v1' or 'v2'")
    return normalize_api_version(raw)


def _expect_dict_response(response: Any, *, context: str) -> Dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    raise ValueError(f"Invalid response from {context}: expected JSON object")


def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _append_query_params(path: str, params: Dict[str, Any]) -> str:
    cleaned = _clean_params(params)
    if not cleaned:
        return path
    return f"{path}?{urlencode(cleaned, doseq=True)}"


def _normalize_algorithm_kind(algorithm_kind: Optional[str]) -> Optional[str]:
    if algorithm_kind is None:
        return None
    normalized = str(algorithm_kind).strip().lower()
    if normalized not in {"gepa", "mipro"}:
        raise ValueError("algorithm_kind must be 'gepa' or 'mipro'")
    return normalized


def _optimizer_event_query_params(
    *,
    limit: int = 200,
    org_id: Optional[str] = None,
    system_id: Optional[str] = None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    rollout_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    seed: Optional[str] = None,
    actor_id: Optional[str] = None,
    stage_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    event_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    algorithm: Optional[str] = None,
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    event_family: Optional[str] = None,
    stream_id: Optional[str] = None,
    trial_id: Optional[str] = None,
    runtime_tick_id: Optional[str] = None,
    proposal_session_id: Optional[str] = None,
    source_session_id: Optional[str] = None,
    sequence: Optional[str] = None,
    source_sequence: Optional[str] = None,
    payload_redacted: Optional[bool] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    q: Optional[str] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    return _clean_params(
        {
            "limit": limit,
            "org_id": org_id,
            "system_id": system_id,
            "job_id": job_id,
            "run_id": run_id,
            "rollout_id": rollout_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "actor_id": actor_id,
            "stage_id": stage_id,
            "trace_id": trace_id,
            "event_id": event_id,
            "causation_id": causation_id,
            "correlation_id": correlation_id,
            "algorithm": algorithm,
            "event_type": event_type,
            "status": status,
            "source": source,
            "event_family": event_family,
            "stream_id": stream_id,
            "trial_id": trial_id,
            "runtime_tick_id": runtime_tick_id,
            "proposal_session_id": proposal_session_id,
            "source_session_id": source_session_id,
            "sequence": sequence,
            "source_sequence": source_sequence,
            "payload_redacted": payload_redacted,
            "start": start,
            "end": end,
            "q": q,
            "cursor": cursor,
        }
    )


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def _parse_429_usage_limit_payload(body: Any) -> UsageLimitError | ValidationError:
    if not isinstance(body, dict):
        return ValidationError(
            "Corrupted HTTP 429 rate-limit payload: payload schema mismatch "
            "(expected JSON object)."
        )
    detail = body.get("detail")
    if not isinstance(detail, dict):
        return ValidationError(
            "Corrupted HTTP 429 rate-limit payload: payload schema mismatch "
            "(detail must be an object with message/limit/current/tier)."
        )

    message = detail.get("message")
    limit = detail.get("limit")
    current = detail.get("current")
    tier = detail.get("tier")
    if not isinstance(message, str):
        return ValidationError(
            "Corrupted HTTP 429 rate-limit payload: payload schema mismatch "
            "(detail.message must be a string)."
        )
    if not isinstance(limit, (int, float)) or not isinstance(current, (int, float)):
        return ValidationError(
            "Corrupted HTTP 429 rate-limit payload: payload schema mismatch "
            "(detail.limit/current must be numeric)."
        )
    if not isinstance(tier, str):
        return ValidationError(
            "Corrupted HTTP 429 rate-limit payload: payload schema mismatch "
            "(detail.tier must be a string)."
        )
    return UsageLimitError(
        limit_type=message,
        api="jobs",
        current=current,
        limit=limit,
        tier=tier,
    )


async def _post_json_with_httpx_fallback(
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{ensure_api_base(base_url).rstrip('/')}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=_auth_headers(api_key))

    if response.status_code == 429:
        parsed: Any
        try:
            parsed = response.json()
        except Exception:
            parsed = {}
        error = _parse_429_usage_limit_payload(parsed)
        raise error

    response.raise_for_status()
    try:
        response_payload = response.json()
    except Exception as exc:
        raise ValidationError(f"Invalid response from {path}: expected JSON object") from exc
    return _expect_dict_response(response_payload, context=path)


async def _patch_state_with_action_canonical(
    *,
    backend_url: str,
    api_key: str,
    timeout: float,
    resource_path: str,
    state: Literal["paused", "running", "cancelled"],
    action: Literal["pause", "resume", "cancel"],
    context: str,
) -> Dict[str, Any]:
    base = ensure_api_base(backend_url).rstrip("/")
    patch_url = f"{base}{resource_path}"
    headers = _auth_headers(api_key)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.patch(
            patch_url,
            json={"state": state},
            headers=headers,
        )
        response.raise_for_status()
        return _expect_dict_response(response.json(), context=context)


def _run_async(coro: Any) -> Any:
    return run_sync(coro, label="Policy optimization client (use async methods in async contexts)")


@dataclass
class PolicyOptimizationSystem:
    """Canonical client for `/{v1|v2}/policy-optimization/systems` resources."""

    system_id: str
    name: str
    technique: str
    backend_url: str
    api_key: str
    timeout: float = 30.0
    api_version: ApiVersion = "v1"

    @classmethod
    async def create_async(
        cls,
        *,
        name: str,
        technique: Literal["discrete_optimization", "model_optimization"] = "discrete_optimization",
        system_id: Optional[str] = None,
        reuse: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> PolicyOptimizationSystem:
        if not name or not name.strip():
            raise ValueError("name is required")

        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        systems_path = policy_systems_base(api_version=version)
        payload: Dict[str, Any] = {
            "name": name.strip(),
            "technique": technique,
            "reuse": bool(reuse),
            "metadata": metadata or {},
        }
        if system_id:
            payload["id"] = system_id

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(systems_path, json=payload)
            data = _expect_dict_response(response, context=systems_path)

        resolved_id = str(data.get("id") or "")
        resolved_name = str(data.get("name") or name.strip())
        resolved_technique = str(data.get("technique") or technique)
        if not resolved_id:
            raise ValueError("Missing id in system response")
        return cls(
            system_id=resolved_id,
            name=resolved_name,
            technique=resolved_technique,
            backend_url=base_url,
            api_key=key,
            timeout=timeout,
            api_version=version,
        )

    @classmethod
    def create(cls, **kwargs: Any) -> PolicyOptimizationSystem:
        return _run_async(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        system_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> PolicyOptimizationSystem:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        system_path = policy_system_path(system_id, api_version=version)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(system_path)
            data = _expect_dict_response(response, context=system_path)
        return cls(
            system_id=str(data.get("id") or system_id),
            name=str(data.get("name") or ""),
            technique=str(data.get("technique") or "discrete_optimization"),
            backend_url=base_url,
            api_key=key,
            timeout=timeout,
            api_version=version,
        )

    @classmethod
    def get(cls, system_id: str, **kwargs: Any) -> PolicyOptimizationSystem:
        return _run_async(cls.get_async(system_id, **kwargs))

    @classmethod
    async def list_async(
        cls,
        *,
        technique: Optional[Literal["discrete_optimization", "model_optimization"]] = None,
        limit: int = 100,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        systems_path = policy_systems_base(api_version=version)
        params = _clean_params({"technique": technique, "limit": limit})
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(systems_path, params=params)
        return _expect_dict_response(response, context=systems_path)

    @classmethod
    def list(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.list_async(**kwargs))

    async def update_async(
        self,
        *,
        name: Optional[str] = None,
        technique: Optional[Literal["discrete_optimization", "model_optimization"]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = _clean_params(
            {
                "name": name.strip() if isinstance(name, str) else None,
                "technique": technique,
                "metadata": metadata,
            }
        )
        if not payload:
            raise ValueError("At least one of name, technique, or metadata must be provided")
        system_path = policy_system_path(self.system_id, api_version=self.api_version)
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{system_path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(url, json=payload, headers=_auth_headers(self.api_key))
            response.raise_for_status()
            data = _expect_dict_response(response.json(), context=system_path)
        self.name = str(data.get("name") or self.name)
        self.technique = str(data.get("technique") or self.technique)
        self.system_id = str(data.get("id") or self.system_id)
        return data

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.update_async(**kwargs))

    async def delete_async(self) -> Dict[str, Any]:
        system_path = policy_system_path(self.system_id, api_version=self.api_version)
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{system_path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, headers=_auth_headers(self.api_key))
            response.raise_for_status()
            if response.content:
                return _expect_dict_response(response.json(), context=system_path)
            return {"id": self.system_id, "archived": True}

    def delete(self) -> Dict[str, Any]:
        return _run_async(self.delete_async())


@dataclass
class PolicyOptimizationOfflineJob:
    """Canonical client for `/{v1|v2}/offline/jobs` resources."""

    job_id: str
    backend_url: str
    api_key: str
    system_id: Optional[str] = None
    system_name: Optional[str] = None
    timeout: float = 30.0
    api_version: ApiVersion = "v1"

    @classmethod
    async def create_async(
        cls,
        *,
        technique: Literal["discrete_optimization"] = "discrete_optimization",
        kind: Literal["gepa_offline", "mipro_offline", "eval"],
        system_name: str,
        system_id: Optional[str] = None,
        reuse_system: bool = True,
        config_mode: Literal["DEFAULT", "FULL"] = "DEFAULT",
        config: Dict[str, Any],
        container_worker_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_start: bool = True,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
        prompt_opt_version: PromptOptVersion = "v1",
        prompt_opt_fallback_policy: Optional[PromptOptFallbackPolicy] = None,
    ) -> PolicyOptimizationOfflineJob:
        if not system_name or not system_name.strip():
            raise ValueError("system_name is required")

        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version or prompt_opt_version)
        jobs_path = offline_jobs_base(api_version=version)

        payload: Dict[str, Any] = {
            "technique": technique,
            "kind": kind,
            "system": {
                "name": system_name.strip(),
                "reuse": bool(reuse_system),
            },
            "config_mode": config_mode,
            "config": config,
            "metadata": metadata or {},
            "auto_start": bool(auto_start),
            "prompt_opt_version": prompt_opt_version,
        }
        if prompt_opt_fallback_policy is not None:
            payload["prompt_opt_fallback_policy"] = prompt_opt_fallback_policy
        if container_worker_token and container_worker_token.strip():
            payload["container_worker_token"] = container_worker_token.strip()
        if system_id:
            payload["system"]["id"] = system_id

        try:
            async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
                response = await http.post_json(jobs_path, json=payload)
                data = _expect_dict_response(response, context=jobs_path)
        except HTTPError as exc:
            if exc.status != 0:
                raise
            data = await _post_json_with_httpx_fallback(
                base_url=base_url,
                api_key=key,
                timeout=timeout,
                path=jobs_path,
                payload=payload,
            )

        job_id = str(data.get("job_id") or "")
        if not job_id:
            raise ValueError("Missing job_id in response")
        system = data.get("system") if isinstance(data.get("system"), dict) else {}
        system_id_value = system.get("id") if isinstance(system, dict) else None
        system_name_value = system.get("name") if isinstance(system, dict) else None

        return cls(
            job_id=job_id,
            backend_url=base_url,
            api_key=key,
            system_id=str(system_id_value) if isinstance(system_id_value, str) else None,
            system_name=str(system_name_value) if isinstance(system_name_value, str) else None,
            timeout=timeout,
            api_version=version,
        )

    @classmethod
    def create(cls, **kwargs: Any) -> PolicyOptimizationOfflineJob:
        return _run_async(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        job_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> PolicyOptimizationOfflineJob:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        job_path = offline_job_path(job_id, api_version=version)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(job_path)
            data = _expect_dict_response(response, context=job_path)
        system = data.get("system") if isinstance(data.get("system"), dict) else {}
        system_id_value = system.get("id") if isinstance(system, dict) else None
        system_name_value = system.get("name") if isinstance(system, dict) else None
        return cls(
            job_id=job_id,
            backend_url=base_url,
            api_key=key,
            system_id=str(system_id_value) if isinstance(system_id_value, str) else None,
            system_name=str(system_name_value) if isinstance(system_name_value, str) else None,
            timeout=timeout,
            api_version=version,
        )

    @classmethod
    def get(cls, job_id: str, **kwargs: Any) -> PolicyOptimizationOfflineJob:
        return _run_async(cls.get_async(job_id, **kwargs))

    @classmethod
    async def list_async(
        cls,
        *,
        state: Optional[str] = None,
        kind: Optional[Literal["gepa_offline", "mipro_offline", "eval"]] = None,
        system_id: Optional[str] = None,
        system_name: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        jobs_path = offline_jobs_base(api_version=version)
        params = _clean_params(
            {
                "state": state,
                "kind": kind,
                "system_id": system_id,
                "system_name": system_name,
                "created_after": created_after,
                "created_before": created_before,
                "limit": limit,
                "cursor": cursor,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(jobs_path, params=params)
        return _expect_dict_response(response, context=jobs_path)

    @classmethod
    def list(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.list_async(**kwargs))

    async def status_async(self) -> Dict[str, Any]:
        job_path = offline_job_path(self.job_id, api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(job_path)
        return _expect_dict_response(response, context=job_path)

    def status(self) -> Dict[str, Any]:
        return _run_async(self.status_async())

    async def events_async(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        events_path = offline_job_path(self.job_id, api_version=self.api_version) + "/events"
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(events_path, params=params)
        return _expect_dict_response(response, context=events_path)

    def events(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        return _run_async(self.events_async(since_seq=since_seq, limit=limit))

    async def artifacts_async(self) -> Any:
        artifacts_path = offline_job_path(self.job_id, api_version=self.api_version) + "/artifacts"
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            return await http.get(artifacts_path)

    def artifacts(self) -> Any:
        return _run_async(self.artifacts_async())

    async def checkpoint_async(self) -> Dict[str, Any]:
        checkpoint_path = (
            offline_job_path(self.job_id, api_version=self.api_version) + "/checkpoint"
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(checkpoint_path)
        return _expect_dict_response(response, context=checkpoint_path)

    def checkpoint(self) -> Dict[str, Any]:
        return _run_async(self.checkpoint_async())

    async def submit_candidates_async(
        self,
        *,
        algorithm_kind: str,
        candidates: list[Dict[str, Any]],
        proposal_session_id: Optional[str] = None,
        proposer_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_algorithm = _normalize_algorithm_kind(algorithm_kind)
        payload: Dict[str, Any] = {
            "job_id": self.job_id,
            "algorithm_kind": normalized_algorithm,
            "candidates": list(candidates or []),
            "proposal_session_id": proposal_session_id,
            "proposer_metadata": dict(proposer_metadata or {}),
        }
        path = candidates_submit_path(api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=path)

    def submit_candidates(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.submit_candidates_async(**kwargs))

    async def get_state_baseline_info_async(self) -> Dict[str, Any]:
        path = offline_job_state_baseline_info_path(self.job_id, api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    def get_state_baseline_info(self) -> Dict[str, Any]:
        return _run_async(self.get_state_baseline_info_async())

    async def get_state_envelope_async(self) -> Dict[str, Any]:
        path = offline_job_state_envelope_path(self.job_id, api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    def get_state_envelope(self) -> Dict[str, Any]:
        return _run_async(self.get_state_envelope_async())

    async def list_trial_queue_async(self) -> Dict[str, Any]:
        path = offline_job_queue_trials_path(self.job_id, api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    def list_trial_queue(self) -> Dict[str, Any]:
        return _run_async(self.list_trial_queue_async())

    async def enqueue_trial_async(
        self,
        *,
        trial: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = _append_query_params(
            offline_job_queue_trials_path(self.job_id, api_version=self.api_version),
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)},
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=dict(trial or {}))
        return _expect_dict_response(response, context=path)

    def enqueue_trial(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.enqueue_trial_async(**kwargs))

    async def update_trial_async(
        self,
        trial_id: str,
        *,
        patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = offline_job_queue_trial_path(
            self.job_id,
            trial_id,
            api_version=self.api_version,
        )
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{path}"
        params = _clean_params(
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)}
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                url,
                params=params,
                json=dict(patch or {}),
                headers=_auth_headers(self.api_key),
            )
            response.raise_for_status()
            return _expect_dict_response(response.json(), context=path)

    def update_trial(self, trial_id: str, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.update_trial_async(trial_id, **kwargs))

    async def cancel_trial_async(
        self,
        trial_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = offline_job_queue_trial_path(
            self.job_id,
            trial_id,
            api_version=self.api_version,
        )
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{path}"
        params = _clean_params(
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)}
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(
                url,
                params=params,
                headers=_auth_headers(self.api_key),
            )
            response.raise_for_status()
            return _expect_dict_response(response.json(), context=path)

    def cancel_trial(
        self,
        trial_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _run_async(
            self.cancel_trial_async(trial_id, algorithm_kind=algorithm_kind)
        )

    async def reorder_trials_async(
        self,
        *,
        trial_ids: list[str],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = _append_query_params(
            offline_job_queue_trials_reorder_path(
                self.job_id,
                api_version=self.api_version,
            ),
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)},
        )
        payload = {"trial_ids": [str(item) for item in trial_ids]}
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=path)

    def reorder_trials(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.reorder_trials_async(**kwargs))

    async def apply_default_trial_plan_async(
        self,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = _append_query_params(
            offline_job_queue_default_plan_path(
                self.job_id,
                api_version=self.api_version,
            ),
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)},
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json={})
        return _expect_dict_response(response, context=path)

    def apply_default_trial_plan(
        self,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _run_async(
            self.apply_default_trial_plan_async(algorithm_kind=algorithm_kind)
        )

    async def get_rollout_queue_async(self) -> Dict[str, Any]:
        path = offline_job_queue_rollouts_path(self.job_id, api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    def get_rollout_queue(self) -> Dict[str, Any]:
        return _run_async(self.get_rollout_queue_async())

    async def set_rollout_queue_policy_async(
        self,
        *,
        policy_patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = offline_job_queue_rollout_policy_path(
            self.job_id,
            api_version=self.api_version,
        )
        params = _clean_params({"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                f"{ensure_api_base(self.backend_url).rstrip('/')}{path}",
                params=params,
                json=dict(policy_patch or {}),
                headers=_auth_headers(self.api_key),
            )
            response.raise_for_status()
            return _expect_dict_response(response.json(), context=path)

    def set_rollout_queue_policy(
        self,
        *,
        policy_patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _run_async(
            self.set_rollout_queue_policy_async(
                policy_patch=policy_patch,
                algorithm_kind=algorithm_kind,
            )
        )

    async def get_rollout_dispatch_metrics_async(self) -> Dict[str, Any]:
        path = offline_job_queue_rollout_metrics_path(
            self.job_id,
            api_version=self.api_version,
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    def get_rollout_dispatch_metrics(self) -> Dict[str, Any]:
        return _run_async(self.get_rollout_dispatch_metrics_async())

    async def get_rollout_limiter_status_async(self) -> Dict[str, Any]:
        path = offline_job_queue_rollout_limiter_status_path(
            self.job_id,
            api_version=self.api_version,
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    def get_rollout_limiter_status(self) -> Dict[str, Any]:
        return _run_async(self.get_rollout_limiter_status_async())

    async def retry_rollout_dispatch_async(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        dispatch_id_norm = str(dispatch_id).strip()
        if not dispatch_id_norm:
            raise ValueError("dispatch_id is required")
        path = _append_query_params(
            offline_job_queue_rollout_retry_path(
                self.job_id,
                dispatch_id_norm,
                api_version=self.api_version,
            ),
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)},
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json={})
        return _expect_dict_response(response, context=path)

    def retry_rollout_dispatch(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _run_async(
            self.retry_rollout_dispatch_async(
                dispatch_id,
                algorithm_kind=algorithm_kind,
            )
        )

    async def drain_rollout_queue_async(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = _append_query_params(
            offline_job_queue_rollout_drain_path(
                self.job_id,
                api_version=self.api_version,
            ),
            {"algorithm_kind": _normalize_algorithm_kind(algorithm_kind)},
        )
        payload = {"cancel_queued": bool(cancel_queued)}
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=path)

    def drain_rollout_queue(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _run_async(
            self.drain_rollout_queue_async(
                cancel_queued=cancel_queued,
                algorithm_kind=algorithm_kind,
            )
        )

    async def pause_async(self) -> Dict[str, Any]:
        return await _patch_state_with_action_canonical(
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            resource_path=offline_job_path(self.job_id, api_version=self.api_version),
            state="paused",
            action="pause",
            context="offline_job.pause",
        )

    def pause(self) -> Dict[str, Any]:
        return _run_async(self.pause_async())

    async def resume_async(self) -> Dict[str, Any]:
        return await _patch_state_with_action_canonical(
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            resource_path=offline_job_path(self.job_id, api_version=self.api_version),
            state="running",
            action="resume",
            context="offline_job.resume",
        )

    def resume(self) -> Dict[str, Any]:
        return _run_async(self.resume_async())

    async def cancel_async(self) -> Dict[str, Any]:
        return await _patch_state_with_action_canonical(
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            resource_path=offline_job_path(self.job_id, api_version=self.api_version),
            state="cancelled",
            action="cancel",
            context="offline_job.cancel",
        )

    def cancel(self) -> Dict[str, Any]:
        return _run_async(self.cancel_async())

    async def restart_from_checkpoint_async(self) -> Dict[str, Any]:
        base = ensure_api_base(self.backend_url).rstrip("/")
        headers = _auth_headers(self.api_key)
        primary = (
            offline_job_path(self.job_id, api_version=self.api_version) + "/restart-from-checkpoint"
        )
        alias = (
            offline_job_path(self.job_id, api_version=self.api_version) + "/restart_from_checkpoint"
        )
        attempted: list[str] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for resource_path in (primary, alias):
                attempted.append(resource_path)
                response = await client.post(f"{base}{resource_path}", headers=headers)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = _expect_dict_response(
                    response.json(),
                    context="offline_job.restart_from_checkpoint",
                )
                # New restart contract: restart may create a child job.
                # Update this client instance to point at the child so subsequent
                # status/events/cancel calls operate on the restarted execution.
                child_job_id = payload.get("child_job_id")
                if isinstance(child_job_id, str) and child_job_id.strip():
                    self.job_id = child_job_id.strip()
                return payload
        raise ValueError(
            "restart_from_checkpoint endpoint not found; attempted: " + ", ".join(attempted)
        )

    def restart_from_checkpoint(self) -> Dict[str, Any]:
        return _run_async(self.restart_from_checkpoint_async())

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 2.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Stream job events via SSE until terminal state.

        Uses Server-Sent Events for real-time delivery with polling fallback.
        """
        from synth_ai.core.streaming import (
            CallbackHandler,
            JobStreamer,
            PromptLearningHandler,
            StreamConfig,
            StreamEndpoints,
            StreamType,
        )

        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            max_events_per_poll=500,
            deduplicate=True,
        )

        effective_handlers: list[Any] = list(handlers) if handlers else [PromptLearningHandler()]
        if on_event is not None:
            effective_handlers.append(CallbackHandler(on_event=on_event))

        streamer = JobStreamer(
            base_url=ensure_api_base(self.backend_url),
            api_key=self.api_key,
            job_id=self.job_id,
            endpoints=StreamEndpoints.offline_job(self.job_id, api_version=self.api_version),
            config=config,
            handlers=effective_handlers,
            interval_seconds=interval,
            timeout_seconds=timeout,
        )

        return await streamer.stream_until_terminal()

    def stream_until_complete(self, **kwargs: Any) -> Dict[str, Any]:
        """Stream job events via SSE until terminal state (sync wrapper)."""
        return _run_async(self.stream_until_complete_async(**kwargs))


@dataclass
class PolicyOptimizationOnlineSession:
    """Canonical client for `/{v1|v2}/online/sessions` resources."""

    session_id: str
    backend_url: str
    api_key: str
    system_id: Optional[str] = None
    system_name: Optional[str] = None
    timeout: float = 30.0
    api_version: ApiVersion = "v1"

    @classmethod
    async def create_async(
        cls,
        *,
        technique: Literal["discrete_optimization"] = "discrete_optimization",
        kind: Literal["gepa_online", "mipro_online", "voyager_online"],
        system_name: str,
        system_id: Optional[str] = None,
        reuse_system: bool = True,
        config_mode: Literal["DEFAULT", "FULL"] = "DEFAULT",
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
        prompt_opt_version: PromptOptVersion = "v1",
        prompt_opt_fallback_policy: Optional[PromptOptFallbackPolicy] = None,
    ) -> PolicyOptimizationOnlineSession:
        if not system_name or not system_name.strip():
            raise ValueError("system_name is required")

        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version or prompt_opt_version)
        sessions_path = online_sessions_base(api_version=version)

        payload: Dict[str, Any] = {
            "technique": technique,
            "kind": kind,
            "system": {
                "name": system_name.strip(),
                "reuse": bool(reuse_system),
            },
            "config_mode": config_mode,
            "config": config,
            "metadata": metadata or {},
            "prompt_opt_version": prompt_opt_version,
        }
        if prompt_opt_fallback_policy is not None:
            payload["prompt_opt_fallback_policy"] = prompt_opt_fallback_policy
        if system_id:
            payload["system"]["id"] = system_id
        if correlation_id:
            payload["correlation_id"] = correlation_id

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(sessions_path, json=payload)
            data = _expect_dict_response(response, context=sessions_path)

        session_id = str(data.get("session_id") or "")
        if not session_id:
            raise ValueError("Missing session_id in response")
        system = data.get("system") if isinstance(data.get("system"), dict) else {}
        system_id_value = system.get("id") if isinstance(system, dict) else None
        system_name_value = system.get("name") if isinstance(system, dict) else None

        return cls(
            session_id=session_id,
            backend_url=base_url,
            api_key=key,
            system_id=str(system_id_value) if isinstance(system_id_value, str) else None,
            system_name=str(system_name_value) if isinstance(system_name_value, str) else None,
            timeout=timeout,
            api_version=version,
        )

    @classmethod
    def create(cls, **kwargs: Any) -> PolicyOptimizationOnlineSession:
        return _run_async(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        session_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> PolicyOptimizationOnlineSession:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        session_path = online_session_path(session_id, api_version=version)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(session_path)
            data = _expect_dict_response(response, context=session_path)
        system = data.get("system") if isinstance(data.get("system"), dict) else {}
        system_id_value = system.get("id") if isinstance(system, dict) else None
        system_name_value = system.get("name") if isinstance(system, dict) else None
        return cls(
            session_id=session_id,
            backend_url=base_url,
            api_key=key,
            system_id=str(system_id_value) if isinstance(system_id_value, str) else None,
            system_name=str(system_name_value) if isinstance(system_name_value, str) else None,
            timeout=timeout,
            api_version=version,
        )

    @classmethod
    def get(cls, session_id: str, **kwargs: Any) -> PolicyOptimizationOnlineSession:
        return _run_async(cls.get_async(session_id, **kwargs))

    @classmethod
    async def list_async(
        cls,
        *,
        state: Optional[str] = None,
        kind: Optional[Literal["gepa_online", "mipro_online", "voyager_online"]] = None,
        system_id: Optional[str] = None,
        system_name: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        sessions_path = online_sessions_base(api_version=version)
        params = _clean_params(
            {
                "state": state,
                "kind": kind,
                "system_id": system_id,
                "system_name": system_name,
                "created_after": created_after,
                "created_before": created_before,
                "limit": limit,
                "cursor": cursor,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(sessions_path, params=params)
        return _expect_dict_response(response, context=sessions_path)

    @classmethod
    def list(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.list_async(**kwargs))

    @classmethod
    async def runtime_compatibility_async(
        cls,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        """Fetch runtime-route compatibility contract.

        This endpoint currently exists on v2 only.
        """
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        if version != "v2":
            raise ValueError("runtime compatibility contract is available only for api_version='v2'")
        path = runtime_compatibility_path(api_version=version)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(path)
        return _expect_dict_response(response, context=path)

    @classmethod
    def runtime_compatibility(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.runtime_compatibility_async(**kwargs))

    @classmethod
    async def runtime_container_rollout_checkpoint_dump_async(
        cls,
        container_id: str,
        rollout_id: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = runtime_container_rollout_checkpoint_dump_path(
            container_id,
            rollout_id,
            api_version=version,
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(path, json=payload or {})
        return _expect_dict_response(response, context=path)

    @classmethod
    def runtime_container_rollout_checkpoint_dump(
        cls,
        container_id: str,
        rollout_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _run_async(
            cls.runtime_container_rollout_checkpoint_dump_async(
                container_id,
                rollout_id,
                **kwargs,
            )
        )

    @classmethod
    async def runtime_container_rollout_checkpoint_restore_async(
        cls,
        container_id: str,
        rollout_id: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = runtime_container_rollout_checkpoint_restore_path(
            container_id,
            rollout_id,
            api_version=version,
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(path, json=payload or {})
        return _expect_dict_response(response, context=path)

    @classmethod
    def runtime_container_rollout_checkpoint_restore(
        cls,
        container_id: str,
        rollout_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _run_async(
            cls.runtime_container_rollout_checkpoint_restore_async(
                container_id,
                rollout_id,
                **kwargs,
            )
        )

    @classmethod
    async def optimizer_events_async(
        cls,
        *,
        limit: int = 200,
        org_id: Optional[str] = None,
        system_id: Optional[str] = None,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        seed: Optional[str] = None,
        actor_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        event_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        event_family: Optional[str] = None,
        stream_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        runtime_tick_id: Optional[str] = None,
        proposal_session_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        sequence: Optional[str] = None,
        source_sequence: Optional[str] = None,
        payload_redacted: Optional[bool] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        q: Optional[str] = None,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = optimizer_events_path(api_version=version)
        params = _optimizer_event_query_params(
            limit=limit,
            org_id=org_id,
            system_id=system_id,
            job_id=job_id,
            run_id=run_id,
            rollout_id=rollout_id,
            candidate_id=candidate_id,
            seed=seed,
            actor_id=actor_id,
            stage_id=stage_id,
            trace_id=trace_id,
            event_id=event_id,
            causation_id=causation_id,
            correlation_id=correlation_id,
            algorithm=algorithm,
            event_type=event_type,
            status=status,
            source=source,
            event_family=event_family,
            stream_id=stream_id,
            trial_id=trial_id,
            runtime_tick_id=runtime_tick_id,
            proposal_session_id=proposal_session_id,
            source_session_id=source_session_id,
            sequence=sequence,
            source_sequence=source_sequence,
            payload_redacted=payload_redacted,
            start=start,
            end=end,
            q=q,
            cursor=cursor,
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    @classmethod
    def optimizer_events(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.optimizer_events_async(**kwargs))

    @classmethod
    async def failure_events_async(
        cls,
        *,
        reason_code: Optional[str] = None,
        error_type: Optional[str] = None,
        limit: int = 200,
        org_id: Optional[str] = None,
        system_id: Optional[str] = None,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        seed: Optional[str] = None,
        actor_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        event_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        event_family: Optional[str] = None,
        stream_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        runtime_tick_id: Optional[str] = None,
        proposal_session_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        sequence: Optional[str] = None,
        source_sequence: Optional[str] = None,
        payload_redacted: Optional[bool] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        q: Optional[str] = None,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = failures_query_path(api_version=version)
        params = _optimizer_event_query_params(
            limit=limit,
            org_id=org_id,
            system_id=system_id,
            job_id=job_id,
            run_id=run_id,
            rollout_id=rollout_id,
            candidate_id=candidate_id,
            seed=seed,
            actor_id=actor_id,
            stage_id=stage_id,
            trace_id=trace_id,
            event_id=event_id,
            causation_id=causation_id,
            correlation_id=correlation_id,
            algorithm=algorithm,
            event_type=event_type,
            status=status,
            source=source,
            event_family=event_family,
            stream_id=stream_id,
            trial_id=trial_id,
            runtime_tick_id=runtime_tick_id,
            proposal_session_id=proposal_session_id,
            source_session_id=source_session_id,
            sequence=sequence,
            source_sequence=source_sequence,
            payload_redacted=payload_redacted,
            start=start,
            end=end,
            q=q,
            cursor=cursor,
        )
        params = _clean_params(
            {
                **params,
                "reason_code": reason_code,
                "error_type": error_type,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    @classmethod
    def failure_events(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.failure_events_async(**kwargs))

    @classmethod
    async def admin_optimizer_events_async(
        cls,
        *,
        limit: int = 200,
        org_id: Optional[str] = None,
        system_id: Optional[str] = None,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        seed: Optional[str] = None,
        actor_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        event_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        event_family: Optional[str] = None,
        stream_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        runtime_tick_id: Optional[str] = None,
        proposal_session_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        sequence: Optional[str] = None,
        source_sequence: Optional[str] = None,
        payload_redacted: Optional[bool] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        q: Optional[str] = None,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = admin_optimizer_events_path(api_version=version)
        params = _optimizer_event_query_params(
            limit=limit,
            org_id=org_id,
            system_id=system_id,
            job_id=job_id,
            run_id=run_id,
            rollout_id=rollout_id,
            candidate_id=candidate_id,
            seed=seed,
            actor_id=actor_id,
            stage_id=stage_id,
            trace_id=trace_id,
            event_id=event_id,
            causation_id=causation_id,
            correlation_id=correlation_id,
            algorithm=algorithm,
            event_type=event_type,
            status=status,
            source=source,
            event_family=event_family,
            stream_id=stream_id,
            trial_id=trial_id,
            runtime_tick_id=runtime_tick_id,
            proposal_session_id=proposal_session_id,
            source_session_id=source_session_id,
            sequence=sequence,
            source_sequence=source_sequence,
            payload_redacted=payload_redacted,
            start=start,
            end=end,
            q=q,
            cursor=cursor,
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    @classmethod
    def admin_optimizer_events(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.admin_optimizer_events_async(**kwargs))

    @classmethod
    async def admin_failure_events_async(
        cls,
        *,
        reason_code: Optional[str] = None,
        error_type: Optional[str] = None,
        limit: int = 200,
        org_id: Optional[str] = None,
        system_id: Optional[str] = None,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        seed: Optional[str] = None,
        actor_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        event_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        event_family: Optional[str] = None,
        stream_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        runtime_tick_id: Optional[str] = None,
        proposal_session_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        sequence: Optional[str] = None,
        source_sequence: Optional[str] = None,
        payload_redacted: Optional[bool] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        q: Optional[str] = None,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = admin_failures_query_path(api_version=version)
        params = _optimizer_event_query_params(
            limit=limit,
            org_id=org_id,
            system_id=system_id,
            job_id=job_id,
            run_id=run_id,
            rollout_id=rollout_id,
            candidate_id=candidate_id,
            seed=seed,
            actor_id=actor_id,
            stage_id=stage_id,
            trace_id=trace_id,
            event_id=event_id,
            causation_id=causation_id,
            correlation_id=correlation_id,
            algorithm=algorithm,
            event_type=event_type,
            status=status,
            source=source,
            event_family=event_family,
            stream_id=stream_id,
            trial_id=trial_id,
            runtime_tick_id=runtime_tick_id,
            proposal_session_id=proposal_session_id,
            source_session_id=source_session_id,
            sequence=sequence,
            source_sequence=source_sequence,
            payload_redacted=payload_redacted,
            start=start,
            end=end,
            q=q,
            cursor=cursor,
        )
        params = _clean_params(
            {
                **params,
                "reason_code": reason_code,
                "error_type": error_type,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    @classmethod
    def admin_failure_events(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.admin_failure_events_async(**kwargs))

    @classmethod
    async def admin_victoria_logs_query_async(
        cls,
        *,
        q: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        cursor: Optional[str] = None,
        redact: Optional[bool] = None,
        limit: int = 200,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion] = None,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        version = _resolve_api_version(api_version)
        path = admin_victoria_logs_query_path(api_version=version)
        params = _clean_params(
            {
                "q": q,
                "start": start,
                "end": end,
                "cursor": cursor,
                "redact": redact,
                "limit": limit,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    @classmethod
    def admin_victoria_logs_query(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.admin_victoria_logs_query_async(**kwargs))

    def _runtime_queue_trials_path(self) -> str:
        if self.system_id:
            return runtime_queue_trials_path(self.system_id, api_version=self.api_version)
        return runtime_session_queue_trials_path(self.session_id, api_version=self.api_version)

    def _runtime_queue_contract_path(self) -> str:
        if self.system_id:
            return runtime_queue_contract_path(self.system_id, api_version=self.api_version)
        return runtime_session_queue_contract_path(self.session_id, api_version=self.api_version)

    def _runtime_queue_trial_path(self, trial_id: str) -> str:
        if self.system_id:
            return runtime_queue_trial_path(self.system_id, trial_id, api_version=self.api_version)
        return runtime_session_queue_trial_path(self.session_id, trial_id, api_version=self.api_version)

    def _runtime_queue_rollouts_path(self) -> str:
        if self.system_id:
            return runtime_queue_rollouts_path(self.system_id, api_version=self.api_version)
        return runtime_session_queue_rollouts_path(self.session_id, api_version=self.api_version)

    def _runtime_queue_rollout_path(self, rollout_id: str) -> str:
        if self.system_id:
            return runtime_queue_rollout_path(self.system_id, rollout_id, api_version=self.api_version)
        return runtime_session_queue_rollout_path(
            self.session_id,
            rollout_id,
            api_version=self.api_version,
        )

    def _runtime_queue_rollout_lease_path(self) -> str:
        if self.system_id:
            return runtime_queue_rollout_lease_path(self.system_id, api_version=self.api_version)
        return runtime_session_queue_rollout_lease_path(self.session_id, api_version=self.api_version)

    def _runtime_queue_rollout_expire_leases_path(self) -> str:
        if self.system_id:
            return runtime_queue_rollout_expire_leases_path(
                self.system_id,
                api_version=self.api_version,
            )
        return runtime_session_queue_rollout_expire_leases_path(
            self.session_id,
            api_version=self.api_version,
        )

    async def runtime_queue_trials_async(
        self,
        *,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = self._runtime_queue_trials_path()
        params = _clean_params(
            {
                "actor": actor,
                "algorithm": algorithm,
                "status": status,
                "limit": limit,
            }
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    def runtime_queue_trials(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_trials_async(**kwargs))

    async def runtime_queue_contract_async(
        self,
        *,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        path = self._runtime_queue_contract_path()
        params = _clean_params({"actor": actor, "algorithm": algorithm})
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    def runtime_queue_contract(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_contract_async(**kwargs))

    async def runtime_queue_patch_contract_async(
        self,
        *,
        queue_contract: Optional[Dict[str, Any]] = None,
        patch: Optional[Dict[str, Any]] = None,
        clear_override: Optional[bool] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        payload = _clean_params(
            {
                "queue_contract": queue_contract,
                "patch": patch,
                "clear_override": clear_override,
            }
        )
        if not payload:
            raise ValueError(
                "contract patch requires at least one of queue_contract, patch, clear_override"
            )
        path = self._runtime_queue_contract_path()
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{path}"
        params = _clean_params(
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            }
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                url,
                params=params,
                json=payload,
                headers=_auth_headers(self.api_key),
            )
            response.raise_for_status()
            return _expect_dict_response(response.json(), context=path)

    def runtime_queue_patch_contract(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_patch_contract_async(**kwargs))

    async def runtime_queue_trial_async(
        self,
        trial_id: str,
        *,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        path = self._runtime_queue_trial_path(trial_id)
        params = _clean_params({"actor": actor, "algorithm": algorithm})
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    def runtime_queue_trial(self, trial_id: str, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_trial_async(trial_id, **kwargs))

    async def runtime_queue_create_trial_async(
        self,
        *,
        candidate_id: str,
        trial_id: Optional[str] = None,
        seed: Optional[int] = None,
        priority: Optional[int] = None,
        checkpoint_ref: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        if not candidate_id or not candidate_id.strip():
            raise ValueError("candidate_id is required")
        base_path = self._runtime_queue_trials_path()
        path = _append_query_params(
            base_path,
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            },
        )
        payload = _clean_params(
            {
                "trial_id": trial_id,
                "candidate_id": candidate_id.strip(),
                "seed": seed,
                "priority": priority,
                "checkpoint_ref": checkpoint_ref,
                "metadata": metadata,
            }
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=base_path)

    def runtime_queue_create_trial(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_create_trial_async(**kwargs))

    async def runtime_queue_patch_trial_async(
        self,
        trial_id: str,
        *,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        cancel: Optional[bool] = None,
        checkpoint_ref: Optional[Dict[str, Any]] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        payload = _clean_params(
            {
                "status": status,
                "priority": priority,
                "cancel": cancel,
                "checkpoint_ref": checkpoint_ref,
            }
        )
        if not payload:
            raise ValueError("patch requires at least one field")
        path = self._runtime_queue_trial_path(trial_id)
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{path}"
        params = _clean_params(
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            }
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                url,
                params=params,
                json=payload,
                headers=_auth_headers(self.api_key),
            )
            response.raise_for_status()
            return _expect_dict_response(response.json(), context=path)

    def runtime_queue_patch_trial(self, trial_id: str, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_patch_trial_async(trial_id, **kwargs))

    async def runtime_queue_rollouts_async(
        self,
        *,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = self._runtime_queue_rollouts_path()
        params = _clean_params(
            {
                "actor": actor,
                "algorithm": algorithm,
                "status": status,
                "limit": limit,
            }
        )
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(path, params=params)
        return _expect_dict_response(response, context=path)

    def runtime_queue_rollouts(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_rollouts_async(**kwargs))

    async def runtime_queue_enqueue_rollout_async(
        self,
        *,
        trial_id: str,
        not_before_ms: Optional[int] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        if not trial_id or not trial_id.strip():
            raise ValueError("trial_id is required")
        base_path = self._runtime_queue_rollouts_path()
        path = _append_query_params(
            base_path,
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            },
        )
        payload = _clean_params({"trial_id": trial_id.strip(), "not_before_ms": not_before_ms})
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=base_path)

    def runtime_queue_enqueue_rollout(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_enqueue_rollout_async(**kwargs))

    async def runtime_queue_lease_rollout_async(
        self,
        *,
        now_ms: Optional[int] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        base_path = self._runtime_queue_rollout_lease_path()
        path = _append_query_params(
            base_path,
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            },
        )
        payload = _clean_params({"now_ms": now_ms})
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=base_path)

    def runtime_queue_lease_rollout(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_lease_rollout_async(**kwargs))

    async def runtime_queue_expire_rollout_leases_async(
        self,
        *,
        now_ms: Optional[int] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        base_path = self._runtime_queue_rollout_expire_leases_path()
        path = _append_query_params(
            base_path,
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            },
        )
        payload = _clean_params({"now_ms": now_ms})
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(path, json=payload)
        return _expect_dict_response(response, context=base_path)

    def runtime_queue_expire_rollout_leases(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_expire_rollout_leases_async(**kwargs))

    async def runtime_queue_patch_rollout_async(
        self,
        rollout_id: str,
        *,
        status: Literal["dispatching", "inflight", "completed", "failed", "cancelled"],
        started_at_ms: Optional[int] = None,
        now_ms: Optional[int] = None,
        expected_state_revision: Optional[int] = None,
        actor: Optional[Literal["runtime", "proposer", "operator"]] = None,
        algorithm: Optional[Literal["gepa", "mipro"]] = None,
    ) -> Dict[str, Any]:
        path = self._runtime_queue_rollout_path(rollout_id)
        payload = _clean_params(
            {
                "status": status,
                "started_at_ms": started_at_ms,
                "now_ms": now_ms,
            }
        )
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}{path}"
        params = _clean_params(
            {
                "actor": actor,
                "algorithm": algorithm,
                "expected_state_revision": expected_state_revision,
            }
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                url,
                params=params,
                json=payload,
                headers=_auth_headers(self.api_key),
            )
            response.raise_for_status()
            return _expect_dict_response(response.json(), context=path)

    def runtime_queue_patch_rollout(self, rollout_id: str, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.runtime_queue_patch_rollout_async(rollout_id, **kwargs))

    async def status_async(self) -> Dict[str, Any]:
        session_path = online_session_path(self.session_id, api_version=self.api_version)
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(session_path)
        return _expect_dict_response(response, context=session_path)

    def status(self) -> Dict[str, Any]:
        return _run_async(self.status_async())

    async def _post_action_async(self, action: str) -> Dict[str, Any]:
        state: Literal["paused", "running", "cancelled"]
        action_name: Literal["pause", "resume", "cancel"]
        if action == "pause":
            state = "paused"
            action_name = "pause"
        elif action == "resume":
            state = "running"
            action_name = "resume"
        elif action == "cancel":
            state = "cancelled"
            action_name = "cancel"
        else:
            raise ValueError("action must be one of pause, resume, cancel")
        return await _patch_state_with_action_canonical(
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            resource_path=online_session_path(self.session_id, api_version=self.api_version),
            state=state,
            action=action_name,
            context=f"online_session.{action}",
        )

    def pause(self) -> Dict[str, Any]:
        return _run_async(self._post_action_async("pause"))

    def resume(self) -> Dict[str, Any]:
        return _run_async(self._post_action_async("resume"))

    def cancel(self) -> Dict[str, Any]:
        return _run_async(self._post_action_async("cancel"))

    async def update_reward_async(
        self,
        *,
        reward_info: Dict[str, Any],
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        trace_ref: Optional[str] = None,
        stop: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"reward_info": reward_info}
        if rollout_id is not None:
            payload["rollout_id"] = rollout_id
        if candidate_id is not None:
            payload["candidate_id"] = candidate_id
        if trace_ref is not None:
            payload["trace_ref"] = trace_ref
        if stop is not None:
            payload["stop"] = bool(stop)
        if metadata is not None:
            payload["metadata"] = metadata

        reward_path = online_session_path(self.session_id, api_version=self.api_version) + "/reward"
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(reward_path, json=payload)
        return _expect_dict_response(response, context=reward_path)

    def update_reward(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.update_reward_async(**kwargs))

    async def events_async(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        events_path = online_session_path(self.session_id, api_version=self.api_version) + "/events"
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(events_path, params=params)
        return _expect_dict_response(response, context=events_path)

    def events(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        return _run_async(self.events_async(since_seq=since_seq, limit=limit))


__all__ = [
    "PolicyOptimizationSystem",
    "PolicyOptimizationOfflineJob",
    "PolicyOptimizationOnlineSession",
]
