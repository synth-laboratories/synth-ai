"""First-class SDK API for reinforcement learning (RL/GSPO).

This module provides high-level abstractions for running RL training jobs
both via CLI (`uvx synth-ai train --type rl`) and programmatically in Python scripts.

Example CLI usage:
    uvx synth-ai train --type rl --config my_config.toml --poll

Example SDK usage:
    from synth_ai.sdk.api.train.rl import RLJob
    from synth_ai.sdk.task.in_process import InProcessTaskApp
    
    async with InProcessTaskApp(task_app_path="my_task_app.py", port=8114) as task_app:
        job = RLJob.from_config(
            config_path="my_config.toml",
            task_app_url=task_app.url,
        )
        job.submit()
        result = job.poll_until_complete()
        print(f"Final reward: {result.get('final_reward', 'N/A')}")
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from synth_ai.core.telemetry import log_info

from .builders import RLBuildResult, build_rl_payload
from .pollers import RLJobPoller
from .task_app import check_task_app_health
from .utils import ensure_api_base, http_post


@dataclass
class RLJobConfig:
    """Configuration for an RL training job."""
    
    config_path: Path
    backend_url: str
    api_key: str
    task_app_url: Optional[str] = None
    task_app_api_key: Optional[str] = None
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    
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
        
        # Get task_app_url from environment if not provided
        if not self.task_app_url:
            self.task_app_url = os.environ.get("TASK_APP_URL")


class RLJob:
    """High-level SDK class for running RL training jobs (GSPO, GRPO, PPO, etc.).
    
    This class provides a clean API for:
    1. Submitting RL training jobs
    2. Polling job status
    3. Retrieving results
    
    Example:
        >>> from synth_ai.sdk.api.train.rl import RLJob
        >>> from synth_ai.sdk.task.in_process import InProcessTaskApp
        >>> 
        >>> # With in-process task app
        >>> async with InProcessTaskApp(
        ...     task_app_path="my_task_app.py",
        ...     port=8114,
        ... ) as task_app:
        ...     job = RLJob.from_config(
        ...         config_path="my_config.toml",
        ...         backend_url="https://api.usesynth.ai",
        ...         api_key=os.environ["SYNTH_API_KEY"],
        ...         task_app_url=task_app.url,
        ...     )
        ...     job_id = job.submit()
        ...     result = job.poll_until_complete(timeout=7200.0)
        ...     print(f"Final reward: {result.get('final_reward', 'N/A')}")
    """
    
    def __init__(
        self,
        config: RLJobConfig,
        job_id: Optional[str] = None,
        skip_health_check: bool = False,
    ) -> None:
        """Initialize an RL training job.

        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
            skip_health_check: If True, skip task app health check before submission.
                              Useful when using tunnels where DNS may not have propagated yet.
        """
        self.config = config
        self._job_id = job_id
        self._build_result: Optional[RLBuildResult] = None
        self._skip_health_check = skip_health_check
    
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_url: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> RLJob:
        """Create an RL job from a config file.
        
        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env var BACKEND_BASE_URL)
            api_key: API key (defaults to env var SYNTH_API_KEY)
            task_app_url: Task app URL (defaults to env var TASK_APP_URL or config file)
            task_app_api_key: Task app API key (defaults to env var ENVIRONMENT_API_KEY)
            allow_experimental: Allow experimental features
            overrides: Config overrides (merged into config)
            idempotency_key: Optional idempotency key for job submission
            
        Returns:
            RLJob instance
            
        Example:
            >>> job = RLJob.from_config(
            ...     config_path="configs/rl_gspo.toml",
            ...     backend_url="https://api.usesynth.ai",
            ...     api_key=os.environ["SYNTH_API_KEY"],
            ...     task_app_url="https://my-task-app.usesynth.ai",
            ... )
        """
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
        
        # Resolve task app URL
        if not task_app_url:
            task_app_url = os.environ.get("TASK_APP_URL")
        
        # Resolve task app API key
        if not task_app_api_key:
            task_app_api_key = os.environ.get("ENVIRONMENT_API_KEY")
        
        config = RLJobConfig(
            config_path=config_path_obj,
            backend_url=backend_url,
            api_key=api_key,
            task_app_url=task_app_url,
            task_app_api_key=task_app_api_key,
            allow_experimental=allow_experimental,
            overrides=overrides,
            idempotency_key=idempotency_key,
        )
        
        return cls(config)
    
    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> RLJob:
        """Resume an existing RL job by ID.
        
        Args:
            job_id: Existing job ID
            backend_url: Backend API URL (defaults to env var BACKEND_BASE_URL)
            api_key: API key (defaults to env var SYNTH_API_KEY)
            
        Returns:
            RLJob instance for the existing job
            
        Example:
            >>> job = RLJob.from_job_id(
            ...     job_id="rl_abc123",
            ...     backend_url="https://api.usesynth.ai",
            ...     api_key=os.environ["SYNTH_API_KEY"],
            ... )
        """
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
        config = RLJobConfig(
            config_path=Path("/dev/null"),  # Dummy path
            backend_url=backend_url,
            api_key=api_key,
        )
        
        return cls(config, job_id=job_id)
    
    def _build_payload(self) -> RLBuildResult:
        """Build the job payload from config."""
        if self._build_result is None:
            if not self.config.config_path.exists() or self.config.config_path.name == "/dev/null":
                raise RuntimeError(
                    "Cannot build payload: config_path is required for new jobs. "
                    "Use from_job_id() to resume an existing job."
                )
            
            overrides = self.config.overrides or {}
            overrides["backend"] = self.config.backend_url
            if self.config.task_app_url:
                overrides["task_url"] = self.config.task_app_url
            
            self._build_result = build_rl_payload(
                config_path=self.config.config_path,
                task_url=self.config.task_app_url or "",
                overrides=overrides,
                idempotency=self.config.idempotency_key,
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
        log_info("RLJob.submit invoked", ctx=ctx)
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check:
            task_app_key = self.config.task_app_api_key or ""
            health = check_task_app_health(build.task_url, task_app_key)
            if not health.ok:
                raise ValueError(f"Task app health check failed: {health.detail}")
        
        # Submit job
        create_url = f"{ensure_api_base(self.config.backend_url)}/rl/jobs"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.config.idempotency_key:
            headers["Idempotency-Key"] = self.config.idempotency_key
        
        # Debug: log the URL being called
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Submitting RL job to: {create_url}")
        
        resp = http_post(create_url, headers=headers, json_body=build.payload)
        
        if resp.status_code not in (200, 201):
            error_msg = f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
            if resp.status_code == 404:
                error_msg += (
                    f"\n\nPossible causes:"
                    f"\n1. Backend route /api/rl/jobs not registered"
                    f"\n2. Backend server needs restart"
                    f"\n3. Verify backend is running at: {self.config.backend_url}"
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
        log_info("RLJob.submit completed", ctx=ctx)
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
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        from synth_ai.sdk.jobs.client import JobsClient
        
        async def _fetch() -> Dict[str, Any]:
            client = JobsClient(
                ensure_api_base(self.config.backend_url),
                self.config.api_key,
            )
            # Use RlJobsApi to get job status
            result = await client.rl.retrieve(job_id=self._job_id)  # type: ignore[arg-type]
            return dict(result) if isinstance(result, dict) else {}
        
        return asyncio.run(_fetch())
    
    def poll_until_complete(
        self,
        *,
        timeout: float = 7200.0,  # Default 2 hours for RL jobs
        interval: float = 10.0,  # Default 10s for RL jobs (longer than prompt learning)
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
            TimeoutError: If timeout exceeded
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        poller = RLJobPoller(
            base_url=ensure_api_base(self.config.backend_url),
            api_key=self.config.api_key,
        )
        
        import time
        start_time = time.time()
        
        while True:
            outcome = poller.poll_job(self._job_id)  # type: ignore[arg-type]
            
            if on_status:
                try:
                    on_status(outcome.payload)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"on_status callback raised exception: {e}")
            
            status = outcome.payload.get("status", "unknown")
            
            # Terminal states
            if status in ("succeeded", "failed", "cancelled"):
                return dict(outcome.payload) if isinstance(outcome.payload, dict) else {}
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Job {self._job_id} did not complete within {timeout}s timeout. "
                    f"Current status: {status}"
                )
            
            time.sleep(interval)
    
    def get_results(self) -> Dict[str, Any]:
        """Get final job results.
        
        Returns:
            Job results dictionary
            
        Raises:
            RuntimeError: If job hasn't completed successfully
        """
        status = self.get_status()
        if status.get("status") != "succeeded":
            raise RuntimeError(
                f"Job not complete: {status.get('status')}. "
                "Call poll_until_complete() first or check job status."
            )
        return status.get("results", {})


__all__ = [
    "RLJob",
    "RLJobConfig",
]

