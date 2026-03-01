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

import httpx

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.optimization_routes import (
    normalize_api_version,
    offline_job_path,
    offline_jobs_base,
    online_session_path,
    online_sessions_base,
    policy_system_path,
    policy_systems_base,
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
    raw = (api_version or os.getenv("SYNTH_POLICY_API_VERSION") or "v2").strip().lower()
    if raw not in {"v1", "v2"}:
        raise ValueError("api_version must be 'v1' or 'v2'")
    return normalize_api_version(raw)


def _expect_dict_response(response: Any, *, context: str) -> Dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    raise ValueError(f"Invalid response from {context}: expected JSON object")


def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
    }


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
    api_version: ApiVersion = "v2"

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
    api_version: ApiVersion = "v2"

    @classmethod
    async def create_async(
        cls,
        *,
        technique: Literal["discrete_optimization"] = "discrete_optimization",
        kind: Literal["gepa_offline", "mipro_offline"],
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
        prompt_opt_version: PromptOptVersion = "v2",
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

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(jobs_path, json=payload)
            data = _expect_dict_response(response, context=jobs_path)

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
    api_version: ApiVersion = "v2"

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
        prompt_opt_version: PromptOptVersion = "v2",
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
