"""Internal prompt learning implementation.

Public API: Use `synth_ai.sdk.optimization.PolicyOptimizationJob` instead.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from synth_ai.core.utils.urls import BACKEND_URL_BASE, is_synthtunnel_url
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.optimization.models import PolicyJobStatus, PromptLearningResult

from .builders import (
    PromptLearningBuildResult,
    build_prompt_learning_payload,
    build_prompt_learning_payload_from_mapping,
)
from .local_api import check_local_api_health
from .pollers import JobPoller, PollOutcome
from .prompt_learning_service import (
    cancel_prompt_learning_job,
    query_prompt_learning_workflow_state,
)
from .utils import ensure_api_base, run_sync

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.prompt_learning.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "PromptLearningJob"):
        raise RuntimeError("Rust core PromptLearningJob required; synth_ai_py is unavailable.")
    return synth_ai_py


def _extract_task_app_url(payload: dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    section = payload
    if isinstance(payload.get("prompt_learning"), dict):
        section = payload.get("prompt_learning", {})
    elif isinstance(payload.get("policy_optimization"), dict):
        section = payload.get("policy_optimization", {})
    if isinstance(section, dict):
        for key in ("task_app_url", "localapi_url", "localapi_url_base"):
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("task_app_url", "localapi_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _infer_task_app_url(config: PromptLearningJobConfig) -> Optional[str]:
    overrides = config.overrides or {}
    for key in ("task_url", "task_app_url"):
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if config.config_dict:
        url = _extract_task_app_url(config.config_dict)
        if url:
            return url
    if config.config_path:
        try:
            payload = synth_ai_py.load_toml(str(config.config_path))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            url = _extract_task_app_url(payload)
            if url:
                return url
    env_url = os.environ.get("TASK_APP_URL", "").strip()
    return env_url or None


@dataclass
class PromptLearningJobConfig:
    """Configuration for a prompt learning job.

    This dataclass holds all the configuration needed to submit and run
    a prompt learning job (GEPA optimization).

    Supports two modes:
    1. **File-based**: Provide `config_path` pointing to a TOML file
    2. **Programmatic**: Provide `config_dict` with the configuration directly

    Both modes go through the same `PromptLearningConfig` Pydantic validation.

    Attributes:
        config_path: Path to the TOML configuration file. Mutually exclusive with config_dict.
        config_dict: Dictionary with prompt learning configuration. Mutually exclusive with config_path.
            Should have the same structure as the TOML file (with 'prompt_learning' section).
        backend_url: Base URL of the Synth API backend (e.g., "https://api.usesynth.ai").
        api_key: Synth API key for authentication.
        task_app_api_key: API key for authenticating with the Local API.
        task_app_worker_token: SynthTunnel worker token for relay auth when using st.usesynth.ai URLs.
        allow_experimental: If True, allows use of experimental models.
        overrides: Dictionary of config overrides.

    Example (file-based):
        >>> config = PromptLearningJobConfig(
        ...     config_path=Path("my_config.toml"),
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ... )

    Example (programmatic):
        >>> config = PromptLearningJobConfig(
        ...     config_dict={
        ...         "prompt_learning": {
        ...             "algorithm": "gepa",
        ...             "task_app_url": "https://tunnel.example.com",
        ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
        ...             "gepa": {...},
        ...         }
        ...     },
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ... )
    """

    backend_url: str
    api_key: str
    config_path: Optional[Path] = None
    config_dict: Optional[Dict[str, Any]] = None
    task_app_api_key: Optional[str] = None
    task_app_worker_token: Optional[str] = field(default=None, repr=False)
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Must provide exactly one of config_path or config_dict
        has_path = self.config_path is not None
        has_dict = self.config_dict is not None

        if has_path and has_dict:
            raise ValueError("Provide either config_path OR config_dict, not both")
        if not has_path and not has_dict:
            raise ValueError("Either config_path or config_dict is required")

        if has_path and not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")

        task_url = _infer_task_app_url(self)
        if task_url and is_synthtunnel_url(task_url):
            if not (self.task_app_worker_token or "").strip():
                raise ValueError(
                    "task_app_worker_token is required for SynthTunnel task_app_url. "
                    "Pass tunnel.worker_token when submitting jobs."
                )
            self.task_app_api_key = None
        else:
            # Get task_app_api_key from environment if not provided
            if not self.task_app_api_key:
                self.task_app_api_key = ensure_localapi_auth(
                    backend_base=self.backend_url,
                    synth_api_key=self.api_key,
                )


class PromptLearningJobPoller(JobPoller):
    """Poller for prompt learning jobs."""

    def poll_job(self, job_id: str) -> PollOutcome:
        """Poll a prompt learning job by ID.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            PollOutcome with status and payload
        """
        return super().poll(f"/api/policy-optimization/online/jobs/{job_id}")


class PromptLearningJob:
    """High-level SDK class for running prompt learning jobs (GEPA).

    This class provides a clean API for:
    1. Submitting prompt learning jobs
    2. Polling job status
    3. Retrieving results

    Example:
        >>> from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
        >>>
        >>> # Create job from config
        >>> job = PromptLearningJob.from_config(
        ...     config_path="my_config.toml",
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key=os.environ["SYNTH_API_KEY"]
        ... )
        >>>
        >>> # Submit job
        >>> job_id = job.submit()
        >>> print(f"Job submitted: {job_id}")
        >>>
        >>> # Poll until complete
        >>> result = job.poll_until_complete(timeout=3600.0)
        >>> print(f"Best score: {result['best_score']}")
        >>>
        >>> # Or poll manually
        >>> status = job.get_status()
        >>> print(f"Status: {status['status']}")
    """

    def __init__(
        self,
        config: PromptLearningJobConfig,
        job_id: Optional[str] = None,
        skip_health_check: bool = False,
    ) -> None:
        """Initialize a prompt learning job.

        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
            skip_health_check: If True, skip task app health check before submission.
                              Useful when using tunnels where DNS may not have propagated yet.
        """
        self.config = config
        self._job_id = job_id
        self._build_result: Optional[PromptLearningBuildResult] = None
        self._skip_health_check = skip_health_check
        self._rust_job: Any | None = None

        rust = _require_rust()
        if job_id:
            self._rust_job = rust.PromptLearningJob.from_job_id(
                job_id, self.config.api_key, self.config.backend_url
            )

    def _ensure_rust_job(self, config_payload: Optional[Dict[str, Any]] = None) -> Any | None:
        rust = _require_rust()
        if self._rust_job is not None:
            return self._rust_job
        if self._job_id:
            self._rust_job = rust.PromptLearningJob.from_job_id(
                self._job_id, self.config.api_key, self.config.backend_url
            )
        elif config_payload is not None:
            self._rust_job = rust.PromptLearningJob.from_dict(
                config_payload,
                self.config.api_key,
                self.config.backend_url,
                self.config.task_app_worker_token,
            )
        return self._rust_job

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        task_app_worker_token: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PromptLearningJob:
        """Create a job from a TOML config file.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            task_app_api_key: Task app API key (defaults to ENVIRONMENT_API_KEY env var)
            task_app_worker_token: SynthTunnel worker token for relay auth
            allow_experimental: Allow experimental models
            overrides: Config overrides

        Returns:
            PromptLearningJob instance

        Raises:
            ValueError: If required config is missing
            FileNotFoundError: If config file doesn't exist
        """

        config_path_obj = Path(config_path)

        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = PromptLearningJobConfig(
            config_path=config_path_obj,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            task_app_worker_token=task_app_worker_token,
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        )

        return cls(config)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        task_app_worker_token: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
        skip_health_check: bool = False,
    ) -> PromptLearningJob:
        """Create a job from a configuration dictionary (programmatic use).

        This allows creating prompt learning jobs without a TOML file, enabling
        programmatic use in notebooks, scripts, and applications.

        The config_dict should have the same structure as a TOML file:
        ```python
        {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "https://...",
                "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                "gepa": {...},
            }
        }
        ```

        Args:
            config_dict: Configuration dictionary with 'prompt_learning' section
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            task_app_api_key: Task app API key (defaults to ENVIRONMENT_API_KEY env var)
            task_app_worker_token: SynthTunnel worker token for relay auth
            allow_experimental: Allow experimental models
            overrides: Config overrides
            skip_health_check: If True, skip task app health check before submission

        Returns:
            PromptLearningJob instance

        Raises:
            ValueError: If required config is missing or invalid

        Example:
            >>> job = PromptLearningJob.from_dict(
            ...     config_dict={
            ...         "prompt_learning": {
            ...             "algorithm": "gepa",
            ...             "task_app_url": "https://tunnel.example.com",
            ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
            ...             "gepa": {
            ...                 "rollout": {"budget": 50, "max_concurrent": 5},
            ...                 "evaluation": {"train_seeds": [1, 2, 3], "val_seeds": [4, 5]},
            ...                 "population": {"num_generations": 2, "children_per_generation": 2},
            ...             },
            ...         }
            ...     },
            ...     api_key="sk_live_...",
            ... )
            >>> job_id = job.submit()
        """

        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = PromptLearningJobConfig(
            config_dict=config_dict,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            task_app_worker_token=task_app_worker_token,
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        )

        # Auto-detect tunnel URLs and skip health check if not explicitly set
        if skip_health_check is False:  # Only auto-detect if not explicitly True
            task_url = config_dict.get("prompt_learning", {}).get(
                "task_app_url"
            ) or config_dict.get("prompt_learning", {}).get("local_api_url")
            if task_url and (
                ".trycloudflare.com" in task_url.lower() or ".cfargotunnel.com" in task_url.lower()
            ):
                skip_health_check = True

        return cls(config, skip_health_check=skip_health_check)

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> PromptLearningJob:
        """Resume an existing job by ID.

        Args:
            job_id: Existing job ID
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            PromptLearningJob instance for the existing job
        """

        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        # Create minimal config (we don't need the config for resuming - use empty dict)
        # The config_dict is never used when resuming since we have the job_id
        config = PromptLearningJobConfig(
            config_dict={"prompt_learning": {"_resumed": True}},  # Placeholder for resume mode
            backend_url=backend_url,
            api_key=api_key,
        )

        return cls(config, job_id=job_id)

    def _build_payload(self) -> PromptLearningBuildResult:
        """Build the job payload from config.

        Supports both file-based (config_path) and programmatic (config_dict) modes.
        Both modes route through the same PromptLearningConfig Pydantic validation.
        """
        if self._build_result is None:
            overrides = self.config.overrides or {}
            overrides["backend"] = self.config.backend_url

            # Route to appropriate builder based on config mode
            if self.config.config_dict is not None:
                # Programmatic mode: use dict-based builder
                self._build_result = build_prompt_learning_payload_from_mapping(
                    raw_config=self.config.config_dict,
                    task_url=None,
                    overrides=overrides,
                    allow_experimental=self.config.allow_experimental,
                    source_label="PromptLearningJob.from_dict",
                )
            elif self.config.config_path is not None:
                # File-based mode: use path-based builder
                if not self.config.config_path.exists():
                    raise RuntimeError(
                        f"Config file not found: {self.config.config_path}. "
                        "Use from_dict() for programmatic config or from_job_id() to resume."
                    )

                self._build_result = build_prompt_learning_payload(
                    config_path=self.config.config_path,
                    task_url=None,
                    overrides=overrides,
                    allow_experimental=self.config.allow_experimental,
                )
            else:
                raise RuntimeError(
                    "Cannot build payload: either config_path or config_dict is required. "
                    "Use from_config() for file-based config, from_dict() for programmatic config, "
                    "or from_job_id() to resume an existing job."
                )
        return self._build_result

    def submit(self) -> str:
        """Submit the job to the backend.

        Returns:
            Job ID

        Raises:
            RuntimeError: If job submission fails
            ValueError: If task app health check fails
        """
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()
        is_synth = is_synthtunnel_url(build.task_url)
        if is_synth and not (self.config.task_app_worker_token or "").strip():
            raise ValueError(
                "task_app_worker_token is required for SynthTunnel task_app_url. "
                "Pass tunnel.worker_token when submitting jobs."
            )

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check:
            if is_synth:
                health = check_local_api_health(
                    build.task_url,
                    "",
                    worker_token=self.config.task_app_worker_token,
                )
            else:
                health = check_local_api_health(build.task_url, self.config.task_app_api_key or "")
            if not health.ok:
                raise ValueError(f"Task app health check failed: {health.detail}")

        # Submit job
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Submitting job to: %s", self.config.backend_url)

        rust_job = self._ensure_rust_job(config_payload=build.payload)
        job_id = rust_job.submit()
        if not job_id:
            raise RuntimeError("Response missing job ID")
        self._job_id = job_id
        return job_id

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        return self._job_id

    async def get_status_async(self) -> Dict[str, Any]:
        """Get current job status (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        result = await asyncio.to_thread(rust_job.get_status)
        return dict(result) if isinstance(result, dict) else {}

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Job status dictionary

        Raises:
            RuntimeError: If job hasn't been submitted yet
            ValueError: If job ID format is invalid
        """
        rust_job = self._ensure_rust_job()
        result = rust_job.get_status()
        return dict(result) if isinstance(result, dict) else {}

    def poll_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        progress: bool = False,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_timeout: float = 180.0,
    ) -> PromptLearningResult:
        """Poll job until it reaches a terminal state.

        Args:
            timeout: Maximum seconds to wait for job completion
            interval: Seconds between poll attempts
            progress: If True, log status updates during polling (useful for notebooks)
            on_status: Optional callback called on each status update (for custom progress handling)
            request_timeout: HTTP timeout for each status request (increase for slow vision tasks)

        Returns:
            PromptLearningResult with typed status, best_score, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout is exceeded

        Example:
            >>> result = job.poll_until_complete(progress=True)
            [00:15] running | score: 0.72
            [00:30] running | score: 0.78
            [00:45] succeeded | score: 0.85
            >>> result.succeeded
            True
            >>> result.best_score
            0.85
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        if not progress and on_status is None:
            result = rust_job.poll_until_complete(timeout, interval)
            payload = dict(result) if isinstance(result, dict) else {}
            return PromptLearningResult.from_response(self._job_id, payload)

        import logging

        logger = logging.getLogger(__name__)
        start_time = time.time()
        last_data: Dict[str, Any] = {}
        error_count = 0
        max_errors = 5

        while time.time() - start_time <= timeout:
            try:
                payload = rust_job.get_status()
                last_data = dict(payload) if isinstance(payload, dict) else {}
                error_count = 0

                if progress:
                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    best_score = (
                        last_data.get("best_score")
                        or last_data.get("best_reward")
                        or last_data.get("best_train_score")
                        or last_data.get("best_train_reward")
                    )
                    score_str = (
                        f"score: {best_score:.2f}" if best_score is not None else "score: --"
                    )
                    iteration = last_data.get("iteration") or last_data.get("current_iteration")
                    iter_str = f" | iter: {iteration}" if iteration is not None else ""
                    status_str = str(last_data.get("status", "pending"))
                    logger.info(
                        "[%02d:%02d] %s | %s%s",
                        mins,
                        secs,
                        status_str,
                        score_str,
                        iter_str,
                    )

                if on_status:
                    on_status(last_data)

                result = PromptLearningResult.from_response(self._job_id, last_data)
                if result.is_terminal:
                    if result.failed:
                        error_msg = (
                            last_data.get("error")
                            or last_data.get("error_message")
                            or last_data.get("failure_reason")
                            or last_data.get("message")
                            or "unknown"
                        )
                        logger.error(
                            "Job %s FAILED — %s",
                            self._job_id,
                            error_msg,
                        )
                        # Dump all available error context
                        for k in (
                            "error",
                            "error_message",
                            "error_details",
                            "failure_reason",
                            "traceback",
                            "message",
                        ):
                            v = last_data.get(k)
                            if v:
                                logger.error("  %s: %s", k, v)
                    elif result.status == PolicyJobStatus.CANCELLED:
                        logger.warning("Job %s was cancelled", self._job_id)
                    else:
                        logger.info("Job %s completed: %s", self._job_id, result.status.value)
                    return result
            except Exception as exc:
                error_count += 1
                logger.warning(
                    "Polling error %s/%s for job %s: %s",
                    error_count,
                    max_errors,
                    self._job_id,
                    exc,
                )
                if error_count >= max_errors:
                    raise RuntimeError(
                        f"Polling failed after {error_count} consecutive errors."
                    ) from exc

            time.sleep(interval)

        if progress:
            logger.warning("Polling timeout after %.0fs for job %s", timeout, self._job_id)

        return PromptLearningResult.from_response(self._job_id, last_data)

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PromptLearningResult:
        """Stream job events until completion using SSE (async)."""
        import contextlib

        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        if rust_job is not None and handlers is None:
            result = await asyncio.to_thread(rust_job.stream_until_complete, timeout)
            payload = dict(result) if isinstance(result, dict) else {}
            if on_event and isinstance(payload, dict):
                with contextlib.suppress(Exception):
                    on_event(payload)
            return PromptLearningResult.from_response(self._job_id, payload)

        from .prompt_learning_streaming import build_prompt_learning_streamer

        streamer = build_prompt_learning_streamer(
            backend_url=ensure_api_base(self.config.backend_url),
            api_key=self.config.api_key,
            job_id=self._job_id,
            handlers=handlers,
            interval=interval,
            timeout=timeout,
        )

        final_status = await streamer.stream_until_terminal()

        if on_event and isinstance(final_status, dict):
            with contextlib.suppress(Exception):
                on_event(final_status)

        # SSE final_status may not have full results - fetch them if job succeeded
        status_str = str(final_status.get("status", "")).lower()
        if status_str in ("succeeded", "completed", "success"):
            with contextlib.suppress(Exception):
                full_results = await self.get_results_async()
                # Merge full results into final_status
                final_status.update(
                    {
                        "best_prompt": full_results.get("best_prompt"),
                        "best_score": full_results.get("best_score"),
                    }
                )

        return PromptLearningResult.from_response(self._job_id, final_status)

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PromptLearningResult:
        """Stream job events until completion using SSE.

        This provides real-time event streaming instead of polling.
        Note: on_event is invoked once with the final status payload; use handlers
        to receive per-event callbacks.
        """
        return run_sync(
            self.stream_until_complete_async(
                timeout=timeout,
                interval=interval,
                handlers=handlers,
                on_event=on_event,
            ),
            label="stream_until_complete() (use stream_until_complete_async in async contexts)",
        )

    async def get_results_async(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.) (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        result = await asyncio.to_thread(rust_job.get_results)
        return dict(result) if isinstance(result, dict) else {}

    def get_results(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.)."""
        rust_job = self._ensure_rust_job()
        result = rust_job.get_results()
        return dict(result) if isinstance(result, dict) else {}

    async def get_best_prompt_text_async(self, rank: int = 1) -> Optional[str]:
        """Get the text of the best prompt by rank (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        if rank < 1:
            raise ValueError(f"Rank must be >= 1, got: {rank}")

        results = await self.get_results_async()
        top_prompts = results.get("top_prompts") or []
        if isinstance(top_prompts, list):
            for prompt_info in top_prompts:
                if not isinstance(prompt_info, dict):
                    continue
                if prompt_info.get("rank") == rank:
                    return prompt_info.get("full_text") or prompt_info.get("prompt")
        return None

    def get_best_prompt_text(self, rank: int = 1) -> Optional[str]:
        """Get the text of the best prompt by rank."""
        return run_sync(
            self.get_best_prompt_text_async(rank=rank),
            label="get_best_prompt_text() (use get_best_prompt_text_async in async contexts)",
        )

    def cancel(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a running prompt learning job.

        Sends a cancellation request to the backend. For GEPA/MiPRO jobs,
        this sets the cancel_requested flag and the optimizer will stop
        at the next checkpoint.

        Args:
            reason: Optional reason for cancellation (recorded in job metadata)

        Returns:
            Dict with cancellation status:
            - job_id: The job ID
            - status: "succeeded", "partial", or "failed"
            - message: Human-readable status message
            - attempt_id: ID of the cancel attempt (for debugging)

        Raises:
            RuntimeError: If job hasn't been submitted yet
            httpx.HTTPStatusError: If the cancellation request fails

        Example:
            >>> job.submit()
            >>> # Later...
            >>> result = job.cancel(reason="No longer needed")
            >>> print(result["message"])
            "Cancellation requested. Job will stop at next checkpoint."
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return cancel_prompt_learning_job(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            reason=reason,
        )

    def query_workflow_state(self) -> Dict[str, Any]:
        """Query the Temporal workflow state for instant polling.

        This queries the workflow directly using its @workflow.query handler,
        providing instant state without database lookups. Useful for real-time
        progress monitoring.

        Returns:
            Dict with workflow state:
            - job_id: The job ID
            - workflow_state: State from the query handler (or None if unavailable)
                - job_id: Job identifier
                - run_id: Current run ID
                - status: Current status (pending, running, succeeded, failed, cancelled)
                - progress: Human-readable progress string
                - algorithm: Algorithm being used (gepa, mipro)
                - error: Error message if failed
            - query_name: Name of the query that was executed
            - error: Error message if query failed (workflow may have completed)

        Raises:
            RuntimeError: If job hasn't been submitted yet

        Example:
            >>> state = job.query_workflow_state()
            >>> if state["workflow_state"]:
            ...     print(f"Status: {state['workflow_state']['status']}")
            ...     print(f"Progress: {state['workflow_state']['progress']}")
            >>> else:
            ...     print(f"Query failed: {state.get('error')}")
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return query_prompt_learning_workflow_state(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
        )


__all__ = [
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    "PromptLearningResult",
]
