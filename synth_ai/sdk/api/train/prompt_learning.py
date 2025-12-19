"""First-class SDK API for prompt learning (MIPRO and GEPA).

This module provides high-level abstractions for running prompt optimization jobs
both via CLI (`uvx synth-ai train`) and programmatically in Python scripts.

Example CLI usage:
    uvx synth-ai train --type prompt_learning --config my_config.toml --poll

Example SDK usage:
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
    
    job = PromptLearningJob.from_config("my_config.toml")
    job.submit()
    result = job.poll_until_complete()
    print(f"Best score: {result['best_score']}")

For domain-specific judging, you can use **Verifier Graphs**. See `PromptLearningJudgeConfig` 
in `synth_ai.sdk.api.train.configs.prompt_learning` for configuration details.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from synth_ai.core.telemetry import log_info

from .builders import PromptLearningBuildResult, build_prompt_learning_payload
from .pollers import JobPoller, PollOutcome
from .task_app import check_task_app_health
from .utils import ensure_api_base, http_post


@dataclass
class PromptLearningJobConfig:
    """Configuration for a prompt learning job."""
    
    config_path: Path
    backend_url: str
    api_key: str
    task_app_api_key: Optional[str] = None
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        
        # Get task_app_api_key from environment if not provided
        if not self.task_app_api_key:
            self.task_app_api_key = os.environ.get("ENVIRONMENT_API_KEY")
            if not self.task_app_api_key:
                raise ValueError(
                    "task_app_api_key is required (provide explicitly or set ENVIRONMENT_API_KEY env var)"
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
        import os
        
        from synth_ai.core.env import get_backend_from_env
        
        config_path_obj = Path(config_path)
        
        # Resolve backend URL
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                base, _ = get_backend_from_env()
                backend_url = f"{base}/api" if not base.endswith("/api") else base
        
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
        import os
        
        from synth_ai.core.env import get_backend_from_env
        
        # Resolve backend URL
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                base, _ = get_backend_from_env()
                backend_url = f"{base}/api" if not base.endswith("/api") else base
        
        # Resolve API key
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )
        
        # Create minimal config (we don't need the config file for resuming)
        config = PromptLearningJobConfig(
            config_path=Path("/dev/null"),  # Dummy path
            backend_url=backend_url,
            api_key=api_key,
        )
        
        return cls(config, job_id=job_id)
    
    def _build_payload(self) -> PromptLearningBuildResult:
        """Build the job payload from config."""
        if self._build_result is None:
            if not self.config.config_path.exists() or self.config.config_path.name == "/dev/null":
                raise RuntimeError(
                    "Cannot build payload: config_path is required for new jobs. "
                    "Use from_job_id() to resume an existing job."
                )
            
            overrides = self.config.overrides or {}
            overrides["backend"] = self.config.backend_url
            
            self._build_result = build_prompt_learning_payload(
                config_path=self.config.config_path,
                task_url=None,  # Force using TOML only
                overrides=overrides,
                allow_experimental=self.config.allow_experimental,
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
        ctx: Dict[str, Any] = {"config_path": str(self.config.config_path)}
        log_info("PromptLearningJob.submit invoked", ctx=ctx)
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check:
            health = check_task_app_health(build.task_url, self.config.task_app_api_key or "")
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
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll job until it reaches a terminal state.
        
        Args:
            timeout: Maximum seconds to wait
            interval: Seconds between poll attempts
            on_status: Optional callback called on each status update
            
        Returns:
            Final job status dictionary
            
        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout is exceeded
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        poller = PromptLearningJobPoller(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            interval=interval,
            timeout=timeout,
        )
        
        outcome = poller.poll_job(self._job_id)  # type: ignore[arg-type]  # We check None above
        
        payload = dict(outcome.payload) if isinstance(outcome.payload, dict) else {}
        
        if on_status:
            on_status(payload)
        
        return payload
    
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
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
]

