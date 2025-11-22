"""First-class SDK API for prompt learning (MIPRO and GEPA).

This module provides high-level abstractions for running prompt optimization jobs
both via CLI (`uvx synth-ai train`) and programmatically in Python scripts.

Example CLI usage:
    uvx synth-ai train --type prompt_learning --config my_config.toml --poll

Example SDK usage:
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    
    job = PromptLearningJob.from_config("my_config.toml")
    job.submit()
    result = job.poll_until_complete()
    print(f"Best score: {result['best_score']}")
"""


import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .builders import PromptLearningBuildResult, build_prompt_learning_payload
from .pollers import JobPoller, PollOutcome
from .task_app import check_task_app_health
from .utils import http_post
from synth_ai.utils import require_keys


@dataclass
class PromptLearningJobConfig:
    """Configuration for a prompt learning job."""
    
    config_path: Path
    synth_key: str
    env_key: str
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None


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
        >>> from synth_ai.api.train.prompt_learning import PromptLearningJob
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
    ) -> None:
        """Initialize a prompt learning job.
        
        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
        """
        self.config = config
        self._job_id = job_id
        self._build_result: Optional[PromptLearningBuildResult] = None
    
    @classmethod
    def from_config(
        cls,
        *,
        config_path: Path,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PromptLearningJob":
        """Create a job from a TOML config file.
        
        Args:
            config_path: Path to TOML config file
            allow_experimental: Allow experimental models
            overrides: Config overrides
            
        Returns:
            PromptLearningJob instance
            
        Raises:
            ValueError: If required config is missing
            FileNotFoundError: If config file doesn't exist
        """


        return cls(PromptLearningJobConfig(
            config_path=config_path,
            synth_key=require_keys("SYNTH_API_KEY")["SYNTH_API_KEY"],
            env_key=require_keys("ENVIRONMENT_API_KEY")["ENVIRONMENT_API_KEY"],
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        ))
    
    @classmethod
    def from_job_id(
        cls,
        job_id: str,
    ) -> "PromptLearningJob":
        """Resume an existing job by ID.
        
        Args:
            job_id: Existing job ID
            
        Returns:
            PromptLearningJob instance for the existing job
        """
        
        # Create minimal config (we don't need the config file for resuming)
        config = PromptLearningJobConfig(
            config_path=Path("/dev/null"),  # Dummy path
            synth_key=require_keys("SYNTH_API_KEY")["SYNTH_API_KEY"],
            env_key=require_keys("ENVIRONMENT_API_KEY")["ENVIRONMENT_API_KEY"],
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
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")
        
        build = self._build_payload()
        
        # Health check
        health = check_task_app_health(build.task_url, self.config.env_key or "")
        if not health.ok:
            raise ValueError(f"Task app health check failed: {health.detail}")
        
        # Submit job
        from synth_ai.urls import BACKEND_PO
        url = f"{BACKEND_PO}/online/jobs"
        headers = {
            "Authorization": f"Bearer {self.config.synth_key}",
            "Content-Type": "application/json",
        }
        
        # Debug: log the URL being called
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Submitting job to: {url}")
        
        resp = http_post(url, headers=headers, json_body=build.payload)
        
        if resp.status_code not in (200, 201):
            error_msg = f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
            raise RuntimeError(error_msg)
        
        try:
            js = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}") from e
        
        job_id = js.get("job_id") or js.get("id")
        if not job_id:
            raise RuntimeError("Response missing job ID")
        
        self._job_id = job_id
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
        
        from synth_ai.learning.prompt_learning_client import PromptLearningClient
        
        async def _fetch() -> Dict[str, Any]:
            client = PromptLearningClient(self.config.synth_key)
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
            api_key=self.config.synth_key,
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
        
        from synth_ai.learning.prompt_learning_client import PromptLearningClient
        
        async def _fetch() -> Dict[str, Any]:
            client = PromptLearningClient(self.config.synth_key)
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
        
        from synth_ai.learning.prompt_learning_client import PromptLearningClient
        
        async def _fetch() -> Optional[str]:
            client = PromptLearningClient(self.config.synth_key)
            return await client.get_prompt_text(self._job_id, rank=rank)  # type: ignore[arg-type]  # We check None above
        
        return asyncio.run(_fetch())


__all__ = [
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
]
