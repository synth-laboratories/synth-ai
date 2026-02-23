"""GEPA online session SDK wrapper.

**Status:** Beta

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

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.optimization.policy.mipro_online_session import (
    _build_session_payload,
    _coerce_str,
    _get_with_canonical,
    _patch_state_with_action_canonical,
    _post_json_with_canonical,
    _resolve_api_key,
    _resolve_backend_url,
)
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync


def _run_async(coro: Any) -> Any:
    return run_sync(coro, label="GepaOnlineSession (use async methods in async contexts)")


def _expect_dict_response(response: Any, *, context: str) -> Dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    raise ValueError(f"Invalid response from {context}: expected JSON object")


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
        warnings.warn(
            'GepaOnlineSession is deprecated and will be removed on 2026-10-01. '
            'Use PolicyOptimizationOnlineSession.create(algorithm="gepa", ...).',
            DeprecationWarning,
            stacklevel=2,
        )
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
        canonical_body = dict(body)
        canonical_body["algorithm"] = "gepa"
        canonical_body.setdefault("technique", "discrete_optimization")
        canonical_body.setdefault(
            "system",
            {"name": session_id or "gepa-online-session"},
        )

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await _post_json_with_canonical(
                http,
                canonical_path="/v1/policy-optimization/online-sessions",
                payload=canonical_body,
            )

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
            result = await _post_json_with_canonical(
                http,
                canonical_path=f"/v1/policy-optimization/online-sessions/{self.session_id}/reward",
                payload=payload,
            )
        return result

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
                f"/v1/policy-optimization/online-sessions/{self.session_id}/prompt",
                params=params or None,
            )
        return _expect_dict_response(result, context="GEPA prompt endpoint")

    def get_prompt_urls(self, *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        return _run_async(self.get_prompt_urls_async(correlation_id=correlation_id))

    async def list_candidates_async(
        self,
        *,
        job_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List canonical candidates for this online GEPA session."""
        resolved_job_id = job_id
        if not resolved_job_id:
            try:
                session_state = await self._get_async()
                if isinstance(session_state, dict):
                    maybe_job_id = session_state.get("job_id")
                    if isinstance(maybe_job_id, str) and maybe_job_id.strip():
                        resolved_job_id = maybe_job_id
            except Exception:
                resolved_job_id = None

        client = PromptLearningClient(
            base_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return await client.list_system_candidates(
            self.session_id,
            job_id=resolved_job_id,
            algorithm=algorithm,
            mode=mode,
            status=status,
            limit=limit,
            cursor=cursor,
            sort=sort,
            include=include,
        )

    def list_candidates(
        self,
        *,
        job_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List canonical candidates for this online GEPA session."""
        return _run_async(
            self.list_candidates_async(
                job_id=job_id,
                algorithm=algorithm,
                mode=mode,
                status=status,
                limit=limit,
                cursor=cursor,
                sort=sort,
                include=include,
            )
        )

    async def get_candidate_async(
        self,
        candidate_id: str,
        *,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a canonical candidate for this online GEPA session."""
        client = PromptLearningClient(
            base_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        resolved_job_id = job_id
        if not resolved_job_id:
            try:
                session_state = await self._get_async()
                if isinstance(session_state, dict):
                    maybe_job_id = session_state.get("job_id")
                    if isinstance(maybe_job_id, str) and maybe_job_id.strip():
                        resolved_job_id = maybe_job_id
            except Exception:
                resolved_job_id = None
        if resolved_job_id:
            return await client.get_candidate(resolved_job_id, candidate_id)

        candidate = await client.get_global_candidate(candidate_id)
        candidate_system_id = str(candidate.get("system_id") or "").strip()
        if candidate_system_id and candidate_system_id != self.session_id:
            raise ValueError(
                f"Candidate {candidate_id!r} does not belong to GEPA session {self.session_id!r}"
            )
        if candidate_system_id == self.session_id:
            return candidate

        cursor: Optional[str] = None
        for _ in range(100):
            page = await client.list_system_candidates(
                self.session_id,
                limit=200,
                cursor=cursor,
            )
            items = page.get("items")
            if not isinstance(items, list) or not items:
                break
            for item in items:
                if isinstance(item, dict) and str(item.get("candidate_id")) == str(candidate_id):
                    return item
            next_cursor = page.get("next_cursor")
            cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
            if cursor is None:
                break
        raise ValueError(
            f"Candidate {candidate_id!r} was not found in GEPA session {self.session_id!r}"
        )

    def get_candidate(
        self,
        candidate_id: str,
        *,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a canonical candidate for this online GEPA session."""
        return _run_async(self.get_candidate_async(candidate_id, job_id=job_id))

    async def list_seed_evals_async(
        self,
        *,
        job_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        success: Optional[bool] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List canonical seed evaluations for this online GEPA session."""
        resolved_job_id = job_id
        if not resolved_job_id:
            try:
                session_state = await self._get_async()
                if isinstance(session_state, dict):
                    maybe_job_id = session_state.get("job_id")
                    if isinstance(maybe_job_id, str) and maybe_job_id.strip():
                        resolved_job_id = maybe_job_id
            except Exception:
                resolved_job_id = None

        client = PromptLearningClient(
            base_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return await client.list_system_seed_evals(
            self.session_id,
            job_id=resolved_job_id,
            candidate_id=candidate_id,
            split=split,
            seed=seed,
            success=success,
            limit=limit,
            cursor=cursor,
            sort=sort,
            include=include,
        )

    def list_seed_evals(
        self,
        *,
        job_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        success: Optional[bool] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List canonical seed evaluations for this online GEPA session."""
        return _run_async(
            self.list_seed_evals_async(
                job_id=job_id,
                candidate_id=candidate_id,
                split=split,
                seed=seed,
                success=success,
                limit=limit,
                cursor=cursor,
                sort=sort,
                include=include,
            )
        )

    def _post_action(self, action: str) -> Dict[str, Any]:
        return _run_async(self._post_action_async(action))

    async def _post_action_async(self, action: str) -> Dict[str, Any]:
        state = {
            "pause": "paused",
            "resume": "running",
            "cancel": "cancelled",
        }.get(action)
        if state is None:
            raise ValueError("action must be one of pause, resume, cancel")
        return await _patch_state_with_action_canonical(
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            session_id=self.session_id,
            state=state,
            action=action,
        )

    def _get(self) -> Dict[str, Any]:
        return _run_async(self._get_async())

    async def _get_async(self) -> Dict[str, Any]:
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            return await _get_with_canonical(
                http,
                canonical_path=f"/v1/policy-optimization/online-sessions/{self.session_id}",
                params={"algorithm": "gepa"},
            )


__all__ = ["GepaOnlineSession"]
