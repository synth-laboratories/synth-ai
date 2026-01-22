"""MIPRO online session SDK wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync
from synth_ai.sdk.shared import AsyncHttpClient


@dataclass
class MiproOnlineSession:
    """Client wrapper for online MIPRO sessions.

    Create a session once, then fetch stable online URLs and post rewards.
    """

    session_id: str
    backend_url: str
    api_key: str
    correlation_id: Optional[str] = None
    proxy_url: Optional[str] = None
    online_url: Optional[str] = None
    chat_completions_url: Optional[str] = None
    timeout: float = 30.0

    @classmethod
    async def create_async(
        cls,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any] | str | Path] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        config_body: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> MiproOnlineSession:
        """Create a new online MIPRO session (async)."""
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)

        body = _build_session_payload(
            config=config,
            config_name=config_name,
            config_path=config_path,
            config_body=config_body,
            overrides=overrides,
            metadata=metadata,
            session_id=session_id,
            correlation_id=correlation_id,
            agent_id=agent_id,
        )

        async with AsyncHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(
                "/prompt-learning/online/mipro/sessions",
                json=body,
            )

        if not isinstance(response, dict):
            raise ValueError("Invalid response from MIPRO session create")
        session_id = str(response.get("session_id") or "")
        if not session_id:
            raise ValueError("Missing session_id in response")

        return cls(
            session_id=session_id,
            backend_url=base_url,
            api_key=key,
            correlation_id=(response.get("correlation_id") or correlation_id or agent_id),
            proxy_url=_coerce_str(response.get("proxy_url")),
            online_url=_coerce_str(response.get("online_url")),
            chat_completions_url=_coerce_str(response.get("chat_completions_url")),
            timeout=timeout,
        )

    @classmethod
    def create(
        cls,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any] | str | Path] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        config_body: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> MiproOnlineSession:
        """Create a new online MIPRO session."""
        return _run_async(
            cls.create_async(
                backend_url=backend_url,
                api_key=api_key,
                config=config,
                config_name=config_name,
                config_path=config_path,
                config_body=config_body,
                overrides=overrides,
                metadata=metadata,
                session_id=session_id,
                correlation_id=correlation_id,
                agent_id=agent_id,
                timeout=timeout,
            )
        )

    @classmethod
    async def get_online_url_async(
        cls,
        session_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        correlation_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> str:
        """Fetch the stable online URL for a session (async)."""
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        params = {}
        if correlation_id:
            params["correlation_id"] = correlation_id

        async with AsyncHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.get(
                f"/prompt-learning/online/mipro/sessions/{session_id}/prompt",
                params=params or None,
            )

        if not isinstance(response, dict):
            raise ValueError("Invalid response from MIPRO prompt endpoint")
        online_url = response.get("online_url")
        if not online_url:
            raise ValueError("Missing online_url in response")
        return str(online_url)

    @classmethod
    def get_online_url(
        cls,
        session_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        correlation_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> str:
        """Fetch the stable online URL for a session."""
        return _run_async(
            cls.get_online_url_async(
                session_id,
                backend_url=backend_url,
                api_key=api_key,
                correlation_id=correlation_id,
                timeout=timeout,
            )
        )

    async def status_async(self) -> Dict[str, Any]:
        """Return session status payload (async)."""
        async with AsyncHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.get(f"/prompt-learning/online/mipro/sessions/{self.session_id}")
        return dict(result) if isinstance(result, dict) else {}

    def status(self) -> Dict[str, Any]:
        """Return session status payload."""
        return _run_async(self.status_async())

    def pause(self) -> Dict[str, Any]:
        return self._post_action("pause")

    def resume(self) -> Dict[str, Any]:
        return self._post_action("resume")

    def cancel(self) -> Dict[str, Any]:
        return self._post_action("cancel")

    async def update_reward_async(
        self,
        *,
        reward_info: Dict[str, Any],
        artifact: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        trace_ref: Optional[str] = None,
        stop: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update reward info for a rollout (trace materialized server-side) (async)."""
        payload: Dict[str, Any] = {
            "reward_info": reward_info,
        }
        if artifact is not None:
            payload["artifact"] = artifact
        if metadata is not None:
            payload["metadata"] = metadata
        if correlation_id or self.correlation_id:
            payload["correlation_id"] = correlation_id or self.correlation_id
        if rollout_id:
            payload["rollout_id"] = rollout_id
        if candidate_id:
            payload["candidate_id"] = candidate_id
        if trace_ref:
            payload["trace_ref"] = trace_ref
        if stop is not None:
            payload["stop"] = stop

        async with AsyncHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.post_json(
                f"/prompt-learning/online/mipro/sessions/{self.session_id}/reward",
                json=payload,
            )
        return dict(result) if isinstance(result, dict) else {}

    def update_reward(
        self,
        *,
        reward_info: Dict[str, Any],
        artifact: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        trace_ref: Optional[str] = None,
        stop: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update reward info for a rollout (trace materialized server-side)."""
        return _run_async(
            self.update_reward_async(
                reward_info=reward_info,
                artifact=artifact,
                metadata=metadata,
                correlation_id=correlation_id,
                rollout_id=rollout_id,
                candidate_id=candidate_id,
                trace_ref=trace_ref,
                stop=stop,
            )
        )

    async def get_prompt_urls_async(
        self, *, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return proxy URLs (responses + chat completions) (async)."""
        params = {}
        if correlation_id or self.correlation_id:
            params["correlation_id"] = correlation_id or self.correlation_id

        async with AsyncHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.get(
                f"/prompt-learning/online/mipro/sessions/{self.session_id}/prompt",
                params=params or None,
            )
        return dict(result) if isinstance(result, dict) else {}

    def get_prompt_urls(self, *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Return proxy URLs (responses + chat completions)."""
        return _run_async(self.get_prompt_urls_async(correlation_id=correlation_id))

    async def _post_action_async(self, action: str) -> Dict[str, Any]:
        async with AsyncHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.post_json(
                f"/prompt-learning/online/mipro/sessions/{self.session_id}/{action}",
                json={},
            )
        return dict(result) if isinstance(result, dict) else {}

    def _post_action(self, action: str) -> Dict[str, Any]:
        return _run_async(self._post_action_async(action))


def _resolve_backend_url(backend_url: Optional[str]) -> str:
    return (backend_url or BACKEND_URL_BASE).rstrip("/")


def _resolve_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    env_key = os.getenv("SYNTH_API_KEY")
    if not env_key:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY env var)")
    return env_key


def _build_session_payload(
    *,
    config: Optional[Dict[str, Any] | str | Path],
    config_name: Optional[str],
    config_path: Optional[str | Path],
    config_body: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
    correlation_id: Optional[str],
    agent_id: Optional[str],
) -> Dict[str, Any]:
    if config is not None and any([config_name, config_path, config_body]):
        raise ValueError("Provide config or config_name/config_path/config_body, not both")

    body: Dict[str, Any] = {
        "overrides": overrides or {},
        "metadata": metadata or {},
    }
    if session_id:
        body["session_id"] = session_id
    if correlation_id:
        body["correlation_id"] = correlation_id
    if agent_id:
        body["agent_id"] = agent_id

    if config is not None:
        if isinstance(config, dict):
            body["config_body"] = config
        else:
            path = Path(config)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            body["config_path"] = str(path)
        return body

    if config_body is not None:
        body["config_body"] = config_body
    if config_name is not None:
        body["config_name"] = config_name
    if config_path is not None:
        body["config_path"] = str(config_path)

    return body


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _run_async(coro: Any) -> Any:
    return run_sync(coro, label="MiproOnlineSession (use async methods in async contexts)")


__all__ = ["MiproOnlineSession"]
