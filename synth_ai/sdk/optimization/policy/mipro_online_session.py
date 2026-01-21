"""MIPRO online session SDK wrapper."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization._impl.utils import ensure_api_base
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

        async def _create() -> Dict[str, Any]:
            async with AsyncHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
                return await http.post_json(
                    "/prompt-learning/online/mipro/sessions",
                    json=body,
                )

        response = _run_async(_create())
        if not isinstance(response, dict):
            raise ValueError("Invalid response from MIPRO session create")
        session_id = str(response.get("session_id") or "")
        if not session_id:
            raise ValueError("Missing session_id in response")

        return cls(
            session_id=session_id,
            backend_url=base_url,
            api_key=key,
            correlation_id=(
                response.get("correlation_id")
                or correlation_id
                or agent_id
            ),
            proxy_url=_coerce_str(response.get("proxy_url")),
            online_url=_coerce_str(response.get("online_url")),
            chat_completions_url=_coerce_str(response.get("chat_completions_url")),
            timeout=timeout,
        )

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
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        params = {}
        if correlation_id:
            params["correlation_id"] = correlation_id

        async def _fetch() -> Dict[str, Any]:
            async with AsyncHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
                return await http.get(
                    f"/prompt-learning/online/mipro/sessions/{session_id}/prompt",
                    params=params or None,
                )

        response = _run_async(_fetch())
        if not isinstance(response, dict):
            raise ValueError("Invalid response from MIPRO prompt endpoint")
        online_url = response.get("online_url")
        if not online_url:
            raise ValueError("Missing online_url in response")
        return str(online_url)

    def status(self) -> Dict[str, Any]:
        """Return session status payload."""
        async def _fetch() -> Dict[str, Any]:
            async with AsyncHttpClient(
                ensure_api_base(self.backend_url),
                self.api_key,
                timeout=self.timeout,
            ) as http:
                return await http.get(
                    f"/prompt-learning/online/mipro/sessions/{self.session_id}"
                )

        result = _run_async(_fetch())
        return dict(result) if isinstance(result, dict) else {}

    def pause(self) -> Dict[str, Any]:
        return self._post_action("pause")

    def resume(self) -> Dict[str, Any]:
        return self._post_action("resume")

    def cancel(self) -> Dict[str, Any]:
        return self._post_action("cancel")

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

        async def _post() -> Dict[str, Any]:
            async with AsyncHttpClient(
                ensure_api_base(self.backend_url),
                self.api_key,
                timeout=self.timeout,
            ) as http:
                return await http.post_json(
                    f"/prompt-learning/online/mipro/sessions/{self.session_id}/reward",
                    json=payload,
                )

        result = _run_async(_post())
        return dict(result) if isinstance(result, dict) else {}

    def get_prompt_urls(
        self, *, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return proxy URLs (responses + chat completions)."""
        params = {}
        if correlation_id or self.correlation_id:
            params["correlation_id"] = correlation_id or self.correlation_id

        async def _fetch() -> Dict[str, Any]:
            async with AsyncHttpClient(
                ensure_api_base(self.backend_url),
                self.api_key,
                timeout=self.timeout,
            ) as http:
                return await http.get(
                    f"/prompt-learning/online/mipro/sessions/{self.session_id}/prompt",
                    params=params or None,
                )

        result = _run_async(_fetch())
        return dict(result) if isinstance(result, dict) else {}

    def _post_action(self, action: str) -> Dict[str, Any]:
        async def _post() -> Dict[str, Any]:
            async with AsyncHttpClient(
                ensure_api_base(self.backend_url),
                self.api_key,
                timeout=self.timeout,
            ) as http:
                return await http.post_json(
                    f"/prompt-learning/online/mipro/sessions/{self.session_id}/{action}",
                    json={},
                )

        result = _run_async(_post())
        return dict(result) if isinstance(result, dict) else {}


def _resolve_backend_url(backend_url: Optional[str]) -> str:
    return (backend_url or BACKEND_URL_BASE).rstrip("/")


def _resolve_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    env_key = os.getenv("SYNTH_API_KEY")
    if not env_key:
        raise ValueError(
            "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
        )
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
    try:
        asyncio.get_running_loop()
        try:
            import nest_asyncio  # type: ignore[unresolved-import]

            nest_asyncio.apply()
            return asyncio.run(coro)
        except ImportError as exc:
            raise RuntimeError(
                "MiproOnlineSession cannot be called from an async context without nest_asyncio."
            ) from exc
    except RuntimeError:
        return asyncio.run(coro)


__all__ = ["MiproOnlineSession"]
