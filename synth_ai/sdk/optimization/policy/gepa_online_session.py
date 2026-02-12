"""GEPA online session SDK wrapper.

This module provides the `GepaOnlineSession` class for managing online GEPA
optimization sessions. In online mode, you drive rollouts locally while the
backend provides prompt candidates through proxy URLs.

Online GEPA workflow:
1. Create a session with your GEPA configuration (prompt_learning.gepa.mode=online)
2. Get proxy URLs for prompt selection
3. Run rollouts locally, calling the proxy URL for each LLM call
4. Report rewards back to the session
5. Backend proposes new prompt candidates based on rewards
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.sdk.optimization.policy.mipro_online_session import (
    _build_session_payload,
    _coerce_str,
    _resolve_api_key,
    _resolve_backend_url,
)
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync


def _run_async(coro: Any) -> Any:
    return run_sync(coro, label="GepaOnlineSession (use async methods in async contexts)")


@dataclass
class GepaOnlineSession:
    """Client wrapper for online GEPA optimization sessions."""

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
    ) -> GepaOnlineSession:
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

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await http.post_json(
                "/prompt-learning/online/gepa/sessions",
                json=body,
            )

        if not isinstance(response, dict):
            raise ValueError("Invalid response from GEPA session create")
        sid = str(response.get("session_id") or "")
        if not sid:
            raise ValueError("Missing session_id in response")

        return cls(
            session_id=sid,
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
    ) -> GepaOnlineSession:
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

    def get_status(self) -> Dict[str, Any]:
        return self._get()

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
        payload: Dict[str, Any] = {"reward_info": reward_info}
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

        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.post_json(
                f"/prompt-learning/online/gepa/sessions/{self.session_id}/reward",
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
        params = {}
        if correlation_id or self.correlation_id:
            params["correlation_id"] = correlation_id or self.correlation_id

        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.get(
                f"/prompt-learning/online/gepa/sessions/{self.session_id}/prompt",
                params=params or None,
            )
        return dict(result) if isinstance(result, dict) else {}

    def get_prompt_urls(self, *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        return _run_async(self.get_prompt_urls_async(correlation_id=correlation_id))

    def _post_action(self, action: str) -> Dict[str, Any]:
        return _run_async(self._post_action_async(action))

    async def _post_action_async(self, action: str) -> Dict[str, Any]:
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.post_json(
                f"/prompt-learning/online/gepa/sessions/{self.session_id}/{action}",
                json={},
            )
        return dict(result) if isinstance(result, dict) else {}

    def _get(self) -> Dict[str, Any]:
        return _run_async(self._get_async())

    async def _get_async(self) -> Dict[str, Any]:
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.get(f"/prompt-learning/online/gepa/sessions/{self.session_id}")
        return dict(result) if isinstance(result, dict) else {}


__all__ = ["GepaOnlineSession"]
