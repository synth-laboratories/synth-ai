"""First-class SDK API for prompt learning (MIPRO and GEPA).

**Status:** Alpha

Note: MIPRO is Experimental, GEPA is Alpha.

This module provides high-level abstractions for running prompt optimization jobs
both via CLI (`uvx synth-ai train`) and programmatically in Python scripts.

Example CLI usage:
    uvx synth-ai train --type prompt_learning --config my_config.toml --poll

Example SDK usage:
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

    job = PromptLearningJob.from_dict(config_dict, api_key="sk_live_...")
    job.submit()
    result = job.poll_until_complete(progress=True)  # Built-in progress printing

    if result.succeeded:
        print(f"Best score: {result.best_score}")
    else:
        print(f"Failed: {result.error}")

For domain-specific verification, you can use **Verifier Graphs**. See `PromptLearningVerifierConfig`
in `synth_ai.sdk.api.train.configs.prompt_learning` for configuration details.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from synth_ai.core.env import PROD_BASE_URL
from synth_ai.core.telemetry import log_info
from synth_ai.sdk.localapi.auth import ensure_localapi_auth


class JobStatus(str, Enum):
    """Status of a prompt learning job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> JobStatus:
        """Convert string to JobStatus, defaulting to PENDING for unknown values."""
        try:
            return cls(status.lower())
        except ValueError:
            return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        """Whether this status is terminal (job won't change further)."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def is_success(self) -> bool:
        """Whether this status indicates success."""
        return self == JobStatus.SUCCEEDED


@dataclass
class PromptLearningResult:
    """Typed result from a prompt learning job.

    Provides clean accessors for common fields instead of raw dict access.

    Example:
        >>> result = job.poll_until_complete()
        >>> if result.succeeded:
        ...     print(f"Best score: {result.best_score}")
        ...     print(f"Best prompt: {result.best_prompt[:100]}...")
        >>> else:
        ...     print(f"Failed: {result.error}")
    """

    job_id: str
    status: JobStatus
    best_score: Optional[float] = None
    best_prompt: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> PromptLearningResult:
        """Create result from API response dict."""
        status_str = data.get("status", "pending")
        status = JobStatus.from_string(status_str)

        # Extract best score from various field names (backward compat)
        best_score = (
            data.get("best_score")
            or data.get("best_reward")
            or data.get("best_train_score")
            or data.get("best_train_reward")
        )

        return cls(
            job_id=job_id,
            status=status,
            best_score=best_score,
            best_prompt=data.get("best_prompt"),
            error=data.get("error"),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        """Whether the job succeeded."""
        return self.status.is_success

    @property
    def failed(self) -> bool:
        """Whether the job failed."""
        return self.status == JobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.status.is_terminal

from .builders import (
    PromptLearningBuildResult,
    build_prompt_learning_payload,
    build_prompt_learning_payload_from_mapping,
)
from .local_api import check_local_api_health
from .pollers import JobPoller, PollOutcome
from .utils import ensure_api_base, http_get, http_post


@dataclass
class PromptLearningJobConfig:
    """Configuration for a prompt learning job.

    This dataclass holds all the configuration needed to submit and run
    a prompt learning job (MIPRO or GEPA optimization).

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
        return super().poll(f"/api/prompt-learning/online/jobs/{job_id}")


class PromptLearningJob:
    """High-level SDK class for running prompt learning jobs (MIPRO or GEPA).
    
    This class provides a clean API for:
    1. Submitting prompt learning jobs
    2. Polling job status
    3. Retrieving results
    
    Example:
        >>> from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
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
        
        # Resolve backend URL - default to production API
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                backend_url = PROD_BASE_URL
        
        # Resolve API key
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

        # Resolve backend URL - default to production API
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                backend_url = PROD_BASE_URL

        # Resolve API key
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
            task_url = config_dict.get("prompt_learning", {}).get("task_app_url") or config_dict.get("prompt_learning", {}).get("local_api_url")
            if task_url and (
                ".trycloudflare.com" in task_url.lower() or 
                ".cfargotunnel.com" in task_url.lower()
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
        
        # Resolve backend URL - default to production API
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                backend_url = PROD_BASE_URL
        
        # Resolve API key
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
            # Pass task_app_api_key to builder via overrides
            if self.config.task_app_api_key:
                overrides["task_app_api_key"] = self.config.task_app_api_key

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
        # Log context based on config mode
        if self.config.config_path is not None:
            ctx: Dict[str, Any] = {"config_path": str(self.config.config_path)}
        else:
            ctx = {"config_mode": "programmatic"}
        log_info("PromptLearningJob.submit invoked", ctx=ctx)
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check:
            health = check_local_api_health(build.task_url, self.config.task_app_api_key or "")
            if not health.ok:
                raise ValueError(f"Task app health check failed: {health.detail}")
        
        # Submit job
        create_url = f"{ensure_api_base(self.config.backend_url)}/prompt-learning/online/jobs"
        headers = {
            "X-API-Key": self.config.api_key,
            "Content-Type": "application/json",
        }
        
        # Debug: log the URL being called
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Submitting job to: {create_url}")
        
        resp = http_post(create_url, headers=headers, json_body=build.payload)
        
        if resp.status_code not in (200, 201):
            error_msg = f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
            if resp.status_code == 404:
                error_msg += (
                    f"\n\nPossible causes:"
                    f"\n1. Backend route /api/prompt-learning/online/jobs not registered"
                    f"\n2. Backend server needs restart (lazy import may have failed)"
                    f"\n3. Check backend logs for: 'Failed to import prompt_learning_online_router'"
                    f"\n4. Verify backend is running at: {self.config.backend_url}"
                )
            raise RuntimeError(error_msg)
        
        try:
            js = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}") from e
        
        job_id = js.get("job_id") or js.get("id")
        if not job_id:
            raise RuntimeError("Response missing job ID")

        self._job_id = job_id
        ctx["job_id"] = job_id
        log_info("PromptLearningJob.submit completed", ctx=ctx)
        return job_id

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        return self._job_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get current job status.
        
        Returns:
            Job status dictionary
            
        Raises:
            RuntimeError: If job hasn't been submitted yet
            ValueError: If job ID format is invalid
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
        
        async def _fetch() -> Dict[str, Any]:
            client = PromptLearningClient(
                ensure_api_base(self.config.backend_url),
                self.config.api_key,
                timeout=30.0,
            )
            result = await client.get_job(self._job_id)  # type: ignore[arg-type]  # We check None above
            return dict(result) if isinstance(result, dict) else {}
        
        return asyncio.run(_fetch())
    
    def poll_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 5.0,
        progress: bool = False,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_timeout: float = 180.0,
    ) -> PromptLearningResult:
        """Poll job until it reaches a terminal state.

        Args:
            timeout: Maximum seconds to wait for job completion
            interval: Seconds between poll attempts
            progress: If True, print status updates during polling (useful for notebooks)
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

        job_id = self._job_id
        base_url = ensure_api_base(self.config.backend_url)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        elapsed = 0.0
        last_data: Dict[str, Any] = {}

        while elapsed <= timeout:
            try:
                # Fetch job status
                url = f"{base_url}/prompt-learning/online/jobs/{job_id}"
                resp = http_get(url, headers=headers, timeout=request_timeout)
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                last_data = dict(data) if isinstance(data, dict) else {}

                status = JobStatus.from_string(last_data.get("status", "pending"))
                best_score = (
                    last_data.get("best_score")
                    or last_data.get("best_reward")
                    or last_data.get("best_train_score")
                    or last_data.get("best_train_reward")
                )

                # Progress output - update on same line to reduce noise
                if progress:
                    mins, secs = divmod(int(elapsed), 60)
                    score_str = f"score: {best_score:.2f}" if best_score is not None else "score: --"
                    iteration = last_data.get("iteration") or last_data.get("current_iteration")
                    iter_str = f" | iter: {iteration}" if iteration is not None else ""
                    # Use \r to update same line, but print newline on terminal state or score change
                    line = f"[{mins:02d}:{secs:02d}] {status.value} | {score_str}{iter_str}"
                    if status.is_terminal or best_score is not None:
                        print(f"\r{line}{'':20}")  # Clear rest of line and newline
                    else:
                        print(f"\r{line}{'':20}", end="", flush=True)

                # Callback for custom handling
                if on_status:
                    on_status(last_data)

                # Check terminal state
                if status.is_terminal:
                    return PromptLearningResult.from_response(job_id, last_data)

            except Exception as exc:
                if progress:
                    print(f"[poll] error: {exc}")
                log_info("poll request failed", ctx={"error": str(exc), "job_id": job_id})

            time.sleep(interval)
            elapsed = time.time() - start_time

        # Timeout reached
        if progress:
            print(f"[poll] timeout after {timeout:.0f}s")

        # Return with whatever data we have, status will indicate not complete
        return PromptLearningResult.from_response(job_id, last_data)
    
    def get_results(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.).
        
        Returns:
            Results dictionary with best_prompt, best_score, etc.
            
        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
        
        async def _fetch() -> Dict[str, Any]:
            client = PromptLearningClient(
                ensure_api_base(self.config.backend_url),
                self.config.api_key,
            )
            results = await client.get_prompts(self._job_id)  # type: ignore[arg-type]  # We check None above
            
            # Convert PromptResults dataclass to dict
            return {
                "best_prompt": results.best_prompt,
                "best_score": results.best_score,
                "top_prompts": results.top_prompts,
                "optimized_candidates": results.optimized_candidates,
                "attempted_candidates": results.attempted_candidates,
                "validation_results": results.validation_results,
            }
        
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an event loop - can't use asyncio.run()
            # Use nest_asyncio to allow nested event loops if available
            try:
                import nest_asyncio  # type: ignore[unresolved-import]
                nest_asyncio.apply()
                return asyncio.run(_fetch())
            except ImportError:
                # Fallback: run the coroutine in the existing loop
                # This requires the caller to be in an async context
                raise RuntimeError(
                    "get_results() cannot be called from an async context. "
                    "Either install nest_asyncio (pip install nest-asyncio) or "
                    "use await get_results_async() instead."
                ) from None
        except RuntimeError:
            # No event loop running - safe to use asyncio.run()
            return asyncio.run(_fetch())
    
    def get_best_prompt_text(self, rank: int = 1) -> Optional[str]:
        """Get the text of the best prompt by rank.
        
        Args:
            rank: Prompt rank (1 = best, 2 = second best, etc.)
            
        Returns:
            Prompt text or None if not found
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
        
        async def _fetch() -> Optional[str]:
            client = PromptLearningClient(
                ensure_api_base(self.config.backend_url),
                self.config.api_key,
            )
            return await client.get_prompt_text(self._job_id, rank=rank)  # type: ignore[arg-type]  # We check None above
        
        return asyncio.run(_fetch())


__all__ = [
    "JobStatus",
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    "PromptLearningResult",
]
