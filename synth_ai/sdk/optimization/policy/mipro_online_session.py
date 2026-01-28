"""MIPRO online session SDK wrapper.

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
    >>> from synth_ai.sdk.optimization.policy import MiproOnlineSession
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization.utils import ensure_api_base, run_sync


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

        async with RustCoreHttpClient(ensure_api_base(base_url), key, timeout=timeout) as http:
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
            result = await http.get(f"/prompt-learning/online/mipro/sessions/{self.session_id}")
        return dict(result) if isinstance(result, dict) else {}

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
            reward_info: Reward information (must include "score" key)
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
                f"/prompt-learning/online/mipro/sessions/{self.session_id}/prompt",
                params=params or None,
            )
        return dict(result) if isinstance(result, dict) else {}

    def get_prompt_urls(self, *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get proxy URLs for prompt selection (synchronous wrapper).

        Synchronous wrapper around `get_prompt_urls_async`. See that method for
        detailed parameter documentation.

        Returns:
            Dictionary with proxy URLs
        """
        return _run_async(self.get_prompt_urls_async(correlation_id=correlation_id))

    async def _post_action_async(self, action: str) -> Dict[str, Any]:
        async with RustCoreHttpClient(
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
