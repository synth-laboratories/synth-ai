"""MIPRO online session SDK wrapper.

**Status:** Beta

This module provides the `MiproOnlineSession` class for managing online MIPRO
optimization sessions. In online mode, you drive rollouts locally while the
backend provides prompt candidates through proxy URLs.

Online MIPRO workflow:
1. Create a session with your MIPRO configuration
2. Get proxy URLs for prompt selection
3. Run rollouts locally, calling the proxy URL for each LLM call
4. Report rewards back to the session
5. Backend generates new prompt proposals based on rewards

Example:
    >>> from synth_ai.recipes import MiproOnlineSession
    >>>
    >>> # Create session
    >>> session = MiproOnlineSession.create(
    ...     config_path="mipro_config.toml",
    ...     api_key=os.environ["SYNTH_API_KEY"]
    ... )
    >>>
    >>> # Get proxy URL for prompt selection
    >>> urls = session.get_prompt_urls()
    >>> proxy_url = urls["online_url"]
    >>>
    >>> # Run rollout: call proxy_url with your task input
    >>> # Proxy selects best prompt candidate and returns LLM response
    >>>
    >>> # Report reward
    >>> session.update_reward(reward_info={"score": 0.85})
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.optimization_routes import (
    MIPRO_API_VERSION,
    online_session_path,
    online_session_subpath,
    online_sessions_base,
)
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync


async def _post_json_with_canonical(
    http: RustCoreHttpClient,
    *,
    canonical_path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    response = await http.post_json(canonical_path, json=payload)
    return _expect_dict_response(response, context=canonical_path)


async def _get_with_canonical(
    http: RustCoreHttpClient,
    *,
    canonical_path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = await http.get(canonical_path, params=params)
    return _expect_dict_response(response, context=canonical_path)


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


async def _patch_state_with_action_canonical(
    *,
    backend_url: str,
    api_key: str,
    timeout: float,
    session_id: str,
    state: str,
    action: str,
) -> Dict[str, Any]:
    base = ensure_api_base(backend_url).rstrip("/")
    canonical_url = f"{base}{online_session_path(session_id, api_version=MIPRO_API_VERSION)}"
    headers = _auth_headers(api_key)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.patch(
            canonical_url,
            json={"state": state},
            headers=headers,
        )
        response.raise_for_status()
        return _expect_dict_response(response.json(), context=f"online_session.{action}")


@dataclass
class MiproOnlineSession:
    """Client wrapper for online MIPRO optimization sessions.

    Manages a single MIPRO online optimization session. In online mode, you
    control the rollout loop locally while the backend provides prompt candidates
    through proxy URLs. This allows real-time prompt evolution as you report
    rewards.

    Key features:
    - **Proxy URLs**: Backend provides URLs that select prompt candidates
    - **Reward reporting**: Report rewards after each rollout
    - **Session management**: Pause, resume, or cancel optimization
    - **Real-time evolution**: Prompts evolve as rewards are reported

    Attributes:
        session_id: Unique session identifier
        backend_url: Backend API base URL
        api_key: Synth API key for authentication
        correlation_id: Optional correlation ID for tracking
        proxy_url: Proxy URL for prompt selection (deprecated, use online_url)
        online_url: Stable proxy URL for prompt selection
        chat_completions_url: URL for chat completions endpoint
        timeout: HTTP request timeout in seconds

    Example:
        >>> session = MiproOnlineSession.create(
        ...     config_path="mipro_config.toml",
        ...     api_key="sk_live_..."
        ... )
        >>>
        >>> # Get proxy URL
        >>> urls = session.get_prompt_urls()
        >>> proxy_url = urls["online_url"]
        >>>
        >>> # Use proxy_url in your rollout loop
        >>> # Report rewards as you go
        >>> session.update_reward(reward_info={"score": 0.9})
    """

    session_id: str
    """Unique session identifier."""
    backend_url: str
    """Backend API base URL."""
    api_key: str
    """Synth API key for authentication."""
    correlation_id: Optional[str] = None
    """Optional correlation ID for tracking rollouts."""
    proxy_url: Optional[str] = None
    """Proxy URL for prompt selection (deprecated, use online_url)."""
    online_url: Optional[str] = None
    """Stable proxy URL for prompt selection."""
    chat_completions_url: Optional[str] = None
    """URL for chat completions endpoint."""
    timeout: float = 30.0
    """HTTP request timeout in seconds."""

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
        """Create a new online MIPRO session (async).

        Creates a new MIPRO online optimization session. The session provides
        proxy URLs that select prompt candidates for your rollouts.

        Args:
            backend_url: Backend API URL (defaults to production)
            api_key: Synth API key (defaults to SYNTH_API_KEY env var)
            config: MIPRO config dict, file path, or Path object
            config_name: Name of config to load from backend
            config_path: Path to TOML config file
            config_body: Config dictionary
            overrides: Config overrides to apply
            metadata: Optional session metadata
            session_id: Optional session ID (for resuming)
            correlation_id: Optional correlation ID for tracking
            agent_id: Optional agent ID (used as correlation_id if provided)
            timeout: HTTP request timeout in seconds

        Returns:
            MiproOnlineSession instance with proxy URLs populated

        Raises:
            ValueError: If config is invalid or response is malformed
            FileNotFoundError: If config_path doesn't exist

        Note:
            Provide exactly one of: config, config_body, config_name, or config_path
        """
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
        canonical_body["kind"] = "mipro_online"
        canonical_body.setdefault("technique", "discrete_optimization")
        canonical_body.setdefault(
            "system",
            {"name": session_id or str(uuid.uuid4())},
        )

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            response = await _post_json_with_canonical(
                http,
                canonical_path=online_sessions_base(api_version=MIPRO_API_VERSION),
                payload=canonical_body,
            )

        session_id = str(response.get("session_id") or "")
        if not session_id:
            raise ValueError("Missing session_id in response")
        normalized_urls = _normalize_prompt_urls(response)

        return cls(
            session_id=session_id,
            backend_url=base_url,
            api_key=key,
            correlation_id=(response.get("correlation_id") or correlation_id or agent_id),
            proxy_url=_coerce_str(normalized_urls.get("proxy_url")),
            online_url=_coerce_str(normalized_urls.get("online_url")),
            chat_completions_url=_coerce_str(normalized_urls.get("chat_completions_url")),
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
        """Create a new online MIPRO session (synchronous wrapper).

        Synchronous wrapper around `create_async`. See `create_async` for
        detailed parameter documentation.

        Returns:
            MiproOnlineSession instance
        """
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
        """Fetch the stable online URL for a session (async).

        Retrieves the proxy URL for prompt selection without creating a full
        session object. Useful for resuming sessions or getting URLs independently.

        Args:
            session_id: Existing session ID
            backend_url: Backend API URL (defaults to production)
            api_key: Synth API key (defaults to SYNTH_API_KEY env var)
            correlation_id: Optional correlation ID
            timeout: HTTP request timeout in seconds

        Returns:
            Proxy URL string for prompt selection

        Raises:
            ValueError: If response is invalid or missing online_url
        """
        base_url = _resolve_backend_url(backend_url)
        key = _resolve_api_key(api_key)
        params = {}
        if correlation_id:
            params["correlation_id"] = correlation_id

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
            status_response = await _get_with_canonical(
                http,
                canonical_path=online_session_path(
                    session_id,
                    api_version=MIPRO_API_VERSION,
                ),
            )
            online_url = _normalize_prompt_urls(status_response).get("online_url")
            if not online_url:
                response = await http.get(
                    online_session_subpath(
                        session_id,
                        "/prompt",
                        api_version=MIPRO_API_VERSION,
                    ),
                    params=params or None,
                )
                status_response = _expect_dict_response(
                    response,
                    context="MIPRO prompt endpoint",
                )
                online_url = _normalize_prompt_urls(status_response).get("online_url")
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
        """Fetch the stable online URL for a session (synchronous wrapper).

        Synchronous wrapper around `get_online_url_async`. See that method for
        detailed parameter documentation.

        Returns:
            Proxy URL string for prompt selection
        """
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
        """Get current session status (async).

        Retrieves the current status of the MIPRO session including optimization
        progress, current candidates, and session metadata.

        Returns:
            Dictionary with session status information
        """
        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            return await _get_with_canonical(
                http,
                canonical_path=online_session_path(
                    self.session_id,
                    api_version=MIPRO_API_VERSION,
                ),
            )

    def status(self) -> Dict[str, Any]:
        """Get current session status (synchronous wrapper).

        Synchronous wrapper around `status_async`.

        Returns:
            Dictionary with session status information
        """
        return _run_async(self.status_async())

    def pause(self) -> Dict[str, Any]:
        """Pause the optimization session.

        Temporarily pauses prompt proposal generation. Rollouts can continue
        but no new proposals will be generated until resumed.

        Returns:
            Dictionary with pause status
        """
        return self._post_action("pause")

    def resume(self) -> Dict[str, Any]:
        """Resume a paused optimization session.

        Resumes prompt proposal generation after a pause.

        Returns:
            Dictionary with resume status
        """
        return self._post_action("resume")

    def cancel(self) -> Dict[str, Any]:
        """Cancel the optimization session.

        Permanently cancels the session. No further proposals will be generated
        and the session cannot be resumed.

        Returns:
            Dictionary with cancellation status
        """
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
        """Report reward for a rollout (async).

        Reports the reward/score for a completed rollout. This feedback is used
        by the backend to generate new prompt proposals. Call this after each
        rollout completes.

        Args:
            reward_info: Reward information. Prefer including a numeric score (e.g. {"score": 0.85}).
                If omitted, the backend may infer reward from `artifact` or `metadata` (when supported),
                for example `metadata={"expected": "...", "predicted": "..."}` for simple classifiers.
            artifact: Optional artifact data from the rollout
            metadata: Optional metadata about the rollout
            correlation_id: Optional correlation ID (overrides session default)
            rollout_id: Optional rollout identifier
            candidate_id: Optional candidate prompt identifier
            trace_ref: Optional trace reference
            stop: Optional flag to stop optimization

        Returns:
            Dictionary with reward acknowledgment

        Example:
            >>> session.update_reward_async(
            ...     reward_info={"score": 0.85, "correct": True},
            ...     rollout_id="rollout_123",
            ...     metadata={"task": "classification"}
            ... )
        """
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

        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await _post_json_with_canonical(
                http,
                canonical_path=online_session_subpath(
                    self.session_id,
                    "/reward",
                    api_version=MIPRO_API_VERSION,
                ),
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
        """Report reward for a rollout (synchronous wrapper).

        Synchronous wrapper around `update_reward_async`. See that method for
        detailed parameter documentation.

        Returns:
            Dictionary with reward acknowledgment
        """
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
        """Get proxy URLs for prompt selection (async).

        Retrieves the proxy URLs that select prompt candidates for your rollouts.
        These URLs should be called with your task inputs - the backend will
        select the best prompt candidate and return the LLM response.

        Args:
            correlation_id: Optional correlation ID (overrides session default)

        Returns:
            Dictionary with "online_url" and optionally "chat_completions_url"

        Example:
            >>> urls = await session.get_prompt_urls_async()
            >>> proxy_url = urls["online_url"]
            >>> # Use proxy_url in your rollout loop
        """
        params = {}
        if correlation_id or self.correlation_id:
            params["correlation_id"] = correlation_id or self.correlation_id

        async with RustCoreHttpClient(
            ensure_api_base(self.backend_url),
            self.api_key,
            timeout=self.timeout,
        ) as http:
            result = await http.get(
                online_session_subpath(
                    self.session_id,
                    "/prompt",
                    api_version=MIPRO_API_VERSION,
                ),
                params=params or None,
            )
        payload = _expect_dict_response(result, context="MIPRO prompt endpoint")
        return _normalize_prompt_urls(payload)

    def get_prompt_urls(self, *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get proxy URLs for prompt selection (synchronous wrapper).

        Synchronous wrapper around `get_prompt_urls_async`. See that method for
        detailed parameter documentation.

        Returns:
            Dictionary with proxy URLs
        """
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
        """List canonical candidates for this online MIPRO session."""
        client = PromptLearningClient(
            base_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            api_version=MIPRO_API_VERSION,
        )
        return await client.list_system_candidates(
            self.session_id,
            job_id=job_id,
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
        """List canonical candidates for this online MIPRO session."""
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
        """Get a canonical candidate for this online MIPRO session."""
        client = PromptLearningClient(
            base_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            api_version=MIPRO_API_VERSION,
        )
        if job_id:
            return await client.get_candidate(job_id, candidate_id)

        candidate = await client.get_global_candidate(candidate_id)
        candidate_system_id = str(candidate.get("system_id") or "").strip()
        if candidate_system_id and candidate_system_id != self.session_id:
            raise ValueError(
                f"Candidate {candidate_id!r} does not belong to MIPRO session {self.session_id!r}"
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
            f"Candidate {candidate_id!r} was not found in MIPRO session {self.session_id!r}"
        )

    def get_candidate(
        self,
        candidate_id: str,
        *,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a canonical candidate for this online MIPRO session."""
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
        """List canonical seed evaluations for this online MIPRO session."""
        client = PromptLearningClient(
            base_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            api_version=MIPRO_API_VERSION,
        )
        return await client.list_system_seed_evals(
            self.session_id,
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
        """List canonical seed evaluations for this online MIPRO session."""
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
    if config is None:
        explicit_sources = [
            config_name is not None,
            config_path is not None,
            config_body is not None,
        ]
        explicit_count = sum(1 for present in explicit_sources if present)
        if explicit_count == 0:
            raise ValueError(
                "Provide exactly one config source: config, config_name, config_path, or config_body"
            )
        if explicit_count > 1:
            raise ValueError(
                "Provide exactly one explicit config source: config_name, config_path, or config_body"
            )

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
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        body["config_path"] = str(path)

    return body


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _expect_dict_response(response: Any, *, context: str) -> Dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    raise ValueError(f"Invalid response from {context}: expected JSON object")


def _normalize_prompt_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    proxy_url = _coerce_str(payload.get("proxy_url"))
    online_url = _coerce_str(payload.get("online_url")) or proxy_url
    chat_completions_url = _coerce_str(payload.get("chat_completions_url"))
    if not chat_completions_url and online_url:
        base = online_url.rstrip("/")
        if base.endswith("/chat/completions"):
            chat_completions_url = base
        else:
            chat_completions_url = f"{base}/chat/completions"
    normalized["proxy_url"] = proxy_url
    normalized["online_url"] = online_url
    normalized["chat_completions_url"] = chat_completions_url
    return normalized


def _run_async(coro: Any) -> Any:
    return run_sync(coro, label="MiproOnlineSession (use async methods in async contexts)")


__all__ = ["MiproOnlineSession"]
