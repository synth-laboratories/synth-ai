"""Internal prompt learning implementation.

Public API: Use `synth_ai.sdk.optimization.PolicyOptimizationJob` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.optimization.models import PromptLearningResult

from .builders import (
    PromptLearningBuildResult,
    build_prompt_learning_payload,
    build_prompt_learning_payload_from_mapping,
)
from .local_api import check_local_api_health
from .pollers import JobPoller, PollOutcome
from .prompt_learning_polling import poll_prompt_learning_until_complete
from .prompt_learning_service import (
    cancel_prompt_learning_job,
    query_prompt_learning_workflow_state,
    submit_prompt_learning_job,
)
from .utils import ensure_api_base, run_sync


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

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PromptLearningJob:
        """Create a job from a TOML config file.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            task_app_api_key: Task app API key (defaults to ENVIRONMENT_API_KEY env var)
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

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check:
            health = check_local_api_health(build.task_url, self.config.task_app_api_key or "")
            if not health.ok:
                raise ValueError(f"Task app health check failed: {health.detail}")

        # Submit job
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Submitting job to: %s", self.config.backend_url)

        js = submit_prompt_learning_job(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            payload=build.payload,
        )

        job_id = js.get("job_id") or js.get("id")
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

        from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import (
            PromptLearningClient,
        )

        client = PromptLearningClient(
            ensure_api_base(self.config.backend_url),
            self.config.api_key,
            timeout=30.0,
        )
        result = await client.get_job(self._job_id)  # type: ignore[arg-type]  # We check None above
        return dict(result) if isinstance(result, dict) else {}

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Job status dictionary

        Raises:
            RuntimeError: If job hasn't been submitted yet
            ValueError: If job ID format is invalid
        """
        return run_sync(
            self.get_status_async(),
            label="get_status() (use get_status_async in async contexts)",
        )

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

        return poll_prompt_learning_until_complete(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            timeout=timeout,
            interval=interval,
            progress=progress,
            on_status=on_status,
            request_timeout=request_timeout,
        )

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

        from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import (
            PromptLearningClient,
        )

        client = PromptLearningClient(
            ensure_api_base(self.config.backend_url),
            self.config.api_key,
        )
        results = await client.get_prompts(self._job_id)  # type: ignore[arg-type]  # We check None above

        return {
            "best_prompt": results.best_prompt,
            "best_score": results.best_score,
            "top_prompts": results.top_prompts,
            "optimized_candidates": results.optimized_candidates,
            "attempted_candidates": results.attempted_candidates,
            "validation_results": results.validation_results,
        }

    def get_results(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.)."""
        return run_sync(
            self.get_results_async(),
            label="get_results() (use get_results_async in async contexts)",
        )

    async def get_best_prompt_text_async(self, rank: int = 1) -> Optional[str]:
        """Get the text of the best prompt by rank (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import (
            PromptLearningClient,
        )

        client = PromptLearningClient(
            ensure_api_base(self.config.backend_url),
            self.config.api_key,
        )
        return await client.get_prompt_text(self._job_id, rank=rank)  # type: ignore[arg-type]  # We check None above

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
