"""Canonical v1 policy optimization SDK clients.

This module exposes explicit offline-job and online-session clients that target
canonical `/v1/policy-optimization/*` routes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import httpx

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync


def _resolve_backend_url(backend_url: Optional[str]) -> str:
    return (backend_url or BACKEND_URL_BASE).rstrip("/")


def _resolve_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    env_key = os.getenv("SYNTH_API_KEY")
    if not env_key:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY env var)")
    return env_key


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
    return run_sync(coro, label="Policy optimization v1 (use async methods in async contexts)")


@dataclass
class PolicyOptimizationSystem:
    """Canonical client for `/v1/policy-optimization/systems` resources."""

    system_id: str
    name: str
    technique: str
    backend_url: str
    api_key: str
    timeout: float = 30.0

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
    ) -> "PolicyOptimizationSystem":
        if not name or not name.strip():
            raise ValueError("name is required")

        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        payload: Dict[str, Any] = {
            "name": name.strip(),
            "technique": technique,
            "reuse": bool(reuse),
            "metadata": metadata or {},
        }
        if system_id:
            payload["id"] = system_id

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json("/v1/policy-optimization/systems", json=payload)
            data = _expect_dict_response(response, context="/v1/policy-optimization/systems")

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
        )

    @classmethod
    def create(cls, **kwargs: Any) -> "PolicyOptimizationSystem":
        return _run_async(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        system_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> "PolicyOptimizationSystem":
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(f"/v1/policy-optimization/systems/{system_id}")
            data = _expect_dict_response(
                response,
                context=f"/v1/policy-optimization/systems/{system_id}",
            )
        return cls(
            system_id=str(data.get("id") or system_id),
            name=str(data.get("name") or ""),
            technique=str(data.get("technique") or "discrete_optimization"),
            backend_url=base_url,
            api_key=key,
            timeout=timeout,
        )

    @classmethod
    def get(cls, system_id: str, **kwargs: Any) -> "PolicyOptimizationSystem":
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
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        params = _clean_params({"technique": technique, "limit": limit})
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get("/v1/policy-optimization/systems", params=params)
        return _expect_dict_response(response, context="/v1/policy-optimization/systems")

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
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}/v1/policy-optimization/systems/{self.system_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(url, json=payload, headers=_auth_headers(self.api_key))
            response.raise_for_status()
            data = _expect_dict_response(
                response.json(),
                context=f"/v1/policy-optimization/systems/{self.system_id}",
            )
        self.name = str(data.get("name") or self.name)
        self.technique = str(data.get("technique") or self.technique)
        self.system_id = str(data.get("id") or self.system_id)
        return data

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.update_async(**kwargs))

    async def delete_async(self) -> Dict[str, Any]:
        url = f"{ensure_api_base(self.backend_url).rstrip('/')}/v1/policy-optimization/systems/{self.system_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, headers=_auth_headers(self.api_key))
            response.raise_for_status()
            if response.content:
                return _expect_dict_response(
                    response.json(),
                    context=f"/v1/policy-optimization/systems/{self.system_id}",
                )
            return {"id": self.system_id, "archived": True}

    def delete(self) -> Dict[str, Any]:
        return _run_async(self.delete_async())


@dataclass
class PolicyOptimizationOfflineJob:
    """Canonical client for `/v1/policy-optimization/offline-jobs` resources."""

    job_id: str
    backend_url: str
    api_key: str
    system_id: Optional[str] = None
    system_name: Optional[str] = None
    timeout: float = 30.0

    @classmethod
    async def create_async(
        cls,
        *,
        technique: Literal["discrete_optimization"] = "discrete_optimization",
        algorithm: Literal["mipro", "gepa"],
        system_name: str,
        system_id: Optional[str] = None,
        reuse_system: bool = True,
        config_mode: Literal["DEFAULT", "FULL"] = "DEFAULT",
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        auto_start: bool = True,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> "PolicyOptimizationOfflineJob":
        if not system_name or not system_name.strip():
            raise ValueError("system_name is required")

        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)

        payload: Dict[str, Any] = {
            "technique": technique,
            "algorithm": algorithm,
            "system": {
                "name": system_name.strip(),
                "reuse": bool(reuse_system),
            },
            "config_mode": config_mode,
            "config": config,
            "metadata": metadata or {},
            "auto_start": bool(auto_start),
        }
        if system_id:
            payload["system"]["id"] = system_id

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json("/v1/policy-optimization/offline-jobs", json=payload)
            data = _expect_dict_response(response, context="/v1/policy-optimization/offline-jobs")

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
        )

    @classmethod
    def create(cls, **kwargs: Any) -> "PolicyOptimizationOfflineJob":
        return _run_async(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        job_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> "PolicyOptimizationOfflineJob":
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(f"/v1/policy-optimization/offline-jobs/{job_id}")
            data = _expect_dict_response(
                response,
                context=f"/v1/policy-optimization/offline-jobs/{job_id}",
            )
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
        )

    @classmethod
    def get(cls, job_id: str, **kwargs: Any) -> "PolicyOptimizationOfflineJob":
        return _run_async(cls.get_async(job_id, **kwargs))

    @classmethod
    async def list_async(
        cls,
        *,
        state: Optional[str] = None,
        algorithm: Optional[Literal["mipro", "gepa"]] = None,
        system_id: Optional[str] = None,
        system_name: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        params = _clean_params(
            {
                "state": state,
                "algorithm": algorithm,
                "system_id": system_id,
                "system_name": system_name,
                "created_after": created_after,
                "created_before": created_before,
                "limit": limit,
                "cursor": cursor,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get("/v1/policy-optimization/offline-jobs", params=params)
        return _expect_dict_response(response, context="/v1/policy-optimization/offline-jobs")

    @classmethod
    def list(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.list_async(**kwargs))

    async def status_async(self) -> Dict[str, Any]:
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(f"/v1/policy-optimization/offline-jobs/{self.job_id}")
        return _expect_dict_response(
            response,
            context=f"/v1/policy-optimization/offline-jobs/{self.job_id}",
        )

    def status(self) -> Dict[str, Any]:
        return _run_async(self.status_async())

    async def events_async(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(
                f"/v1/policy-optimization/offline-jobs/{self.job_id}/events",
                params=params,
            )
        return _expect_dict_response(
            response,
            context=f"/v1/policy-optimization/offline-jobs/{self.job_id}/events",
        )

    def events(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        return _run_async(self.events_async(since_seq=since_seq, limit=limit))

    async def artifacts_async(self) -> Any:
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            return await http.get(f"/v1/policy-optimization/offline-jobs/{self.job_id}/artifacts")

    def artifacts(self) -> Any:
        return _run_async(self.artifacts_async())

    async def pause_async(self) -> Dict[str, Any]:
        return await _patch_state_with_action_canonical(
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            resource_path=f"/v1/policy-optimization/offline-jobs/{self.job_id}",
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
            resource_path=f"/v1/policy-optimization/offline-jobs/{self.job_id}",
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
            resource_path=f"/v1/policy-optimization/offline-jobs/{self.job_id}",
            state="cancelled",
            action="cancel",
            context="offline_job.cancel",
        )

    def cancel(self) -> Dict[str, Any]:
        return _run_async(self.cancel_async())


@dataclass
class PolicyOptimizationOnlineSession:
    """Canonical client for `/v1/policy-optimization/online-sessions` resources."""

    session_id: str
    backend_url: str
    api_key: str
    system_id: Optional[str] = None
    system_name: Optional[str] = None
    timeout: float = 30.0

    @classmethod
    async def create_async(
        cls,
        *,
        technique: Literal["discrete_optimization"] = "discrete_optimization",
        algorithm: Literal["mipro", "gepa"],
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
    ) -> "PolicyOptimizationOnlineSession":
        if not system_name or not system_name.strip():
            raise ValueError("system_name is required")

        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)

        payload: Dict[str, Any] = {
            "technique": technique,
            "algorithm": algorithm,
            "system": {
                "name": system_name.strip(),
                "reuse": bool(reuse_system),
            },
            "config_mode": config_mode,
            "config": config,
            "metadata": metadata or {},
        }
        if system_id:
            payload["system"]["id"] = system_id
        if correlation_id:
            payload["correlation_id"] = correlation_id

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json("/v1/policy-optimization/online-sessions", json=payload)
            data = _expect_dict_response(response, context="/v1/policy-optimization/online-sessions")

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
        )

    @classmethod
    def create(cls, **kwargs: Any) -> "PolicyOptimizationOnlineSession":
        return _run_async(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        session_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> "PolicyOptimizationOnlineSession":
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(f"/v1/policy-optimization/online-sessions/{session_id}")
            data = _expect_dict_response(
                response,
                context=f"/v1/policy-optimization/online-sessions/{session_id}",
            )
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
        )

    @classmethod
    def get(cls, session_id: str, **kwargs: Any) -> "PolicyOptimizationOnlineSession":
        return _run_async(cls.get_async(session_id, **kwargs))

    @classmethod
    async def list_async(
        cls,
        *,
        state: Optional[str] = None,
        algorithm: Optional[Literal["mipro", "gepa"]] = None,
        system_id: Optional[str] = None,
        system_name: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        params = _clean_params(
            {
                "state": state,
                "algorithm": algorithm,
                "system_id": system_id,
                "system_name": system_name,
                "created_after": created_after,
                "created_before": created_before,
                "limit": limit,
                "cursor": cursor,
            }
        )
        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get("/v1/policy-optimization/online-sessions", params=params)
        return _expect_dict_response(response, context="/v1/policy-optimization/online-sessions")

    @classmethod
    def list(cls, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(cls.list_async(**kwargs))

    async def status_async(self) -> Dict[str, Any]:
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(f"/v1/policy-optimization/online-sessions/{self.session_id}")
        return _expect_dict_response(
            response,
            context=f"/v1/policy-optimization/online-sessions/{self.session_id}",
        )

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
            resource_path=f"/v1/policy-optimization/online-sessions/{self.session_id}",
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

        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.post_json(
                f"/v1/policy-optimization/online-sessions/{self.session_id}/reward",
                json=payload,
            )
        return _expect_dict_response(
            response,
            context=f"/v1/policy-optimization/online-sessions/{self.session_id}/reward",
        )

    def update_reward(self, **kwargs: Any) -> Dict[str, Any]:
        return _run_async(self.update_reward_async(**kwargs))

    async def events_async(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            response = await http.get(
                f"/v1/policy-optimization/online-sessions/{self.session_id}/events",
                params=params,
            )
        return _expect_dict_response(
            response,
            context=f"/v1/policy-optimization/online-sessions/{self.session_id}/events",
        )

    def events(self, *, since_seq: int = 0, limit: int = 500) -> Dict[str, Any]:
        return _run_async(self.events_async(since_seq=since_seq, limit=limit))


__all__ = [
    "PolicyOptimizationSystem",
    "PolicyOptimizationOfflineJob",
    "PolicyOptimizationOnlineSession",
]
