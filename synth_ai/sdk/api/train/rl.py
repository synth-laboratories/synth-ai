"""First-class SDK API for reinforcement learning (RL/GSPO).

**Status:** Experimental

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
            localapi_url=task_app.url,
        )
        job.submit()
        result = job.poll_until_complete()
        print(f"Final reward: {result.get('final_reward', 'N/A')}")
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from synth_ai.core.telemetry import log_info
from synth_ai.core.urls import synth_api_base, synth_rl_jobs_url
from synth_ai.sdk.localapi.auth import ensure_localapi_auth

from .builders import RLBuildResult, build_rl_payload
from .local_api import check_local_api_health
from .pollers import RLJobPoller
from .utils import http_post


@dataclass
class RLJobConfig:
    """Configuration for an RL training job.

    This dataclass holds all the configuration needed to submit and run
    a reinforcement learning training job (GSPO, GRPO, PPO, etc.).

    Attributes:
        config_path: Path to the TOML configuration file that defines the
            RL training task, including model settings, training hyperparameters,
            reward configuration, and Local API URL.
        synth_base_url: Base URL of the Synth API backend (e.g.,
            "https://api.usesynth.ai"). Can also be set via SYNTH_BACKEND_URL.
        synth_user_key: Synth API key for authentication. Can also be set via
            SYNTH_API_KEY environment variable.
        localapi_url: URL of the LocalAPI that serves rollout environments.
            Can be set via SYNTH_LOCALAPI_URL env var if not provided.
        localapi_key: API key for authenticating with the LocalAPI.
            Defaults to ENVIRONMENT_API_KEY env var if not provided.
        allow_experimental: If True, allows use of experimental models and
            features. Defaults to None (uses config file setting).
        overrides: Dictionary of config overrides that take precedence over
            values in the TOML file. Useful for programmatic customization.
        idempotency_key: Optional key for idempotent job submission. If provided,
            submitting the same key twice will return the existing job instead
            of creating a new one.

    Example:
        >>> config = RLJobConfig(
        ...     config_path=Path("rl_config.toml"),
        ...     synth_base_url="https://api.usesynth.ai",
        ...     synth_user_key="sk_live_...",
        ...     localapi_url="https://my-task-app.example.com",
        ... )
    """

    config_path: Path
    synth_user_key: str
    localapi_url: Optional[str] = None
    localapi_key: Optional[str] = None
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    synth_base_url: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not self.synth_user_key:
            raise ValueError("synth_user_key is required")

        # Get localapi_key from environment if not provided
        if not self.localapi_key:
            self.localapi_key = ensure_localapi_auth(
                synth_user_key=self.synth_user_key,
                synth_base_url=self.synth_base_url,
            )

        # Get localapi_url from environment if not provided
        if not self.localapi_url:
            self.localapi_url = os.environ.get("SYNTH_LOCALAPI_URL")


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
        >>> # With in-process LocalAPI
        >>> async with InProcessTaskApp(
        ...     task_app_path="my_task_app.py",
        ...     port=8114,
        ... ) as task_app:
        ...     job = RLJob.from_config(
        ...         config_path="my_config.toml",
        ...         synth_base_url="https://api.usesynth.ai",
        ...         synth_user_key=os.environ["SYNTH_API_KEY"],
        ...         localapi_url=task_app.url,
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
            skip_health_check: If True, skip LocalAPI health check before submission.
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
        synth_user_key: Optional[str] = None,
        localapi_url: Optional[str] = None,
        localapi_key: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        synth_base_url: Optional[str] = None,
    ) -> "RLJob":
        """Create an RL job from a config file.

        Args:
            config_path: Path to TOML config file
            synth_base_url: Backend API URL (defaults to SYNTH_BACKEND_URL)
            synth_user_key: API key (defaults to env var SYNTH_API_KEY)
            localapi_url: LocalAPI URL (defaults to env var SYNTH_LOCALAPI_URL or config file)
            localapi_key: LocalAPI key (defaults to env var ENVIRONMENT_API_KEY)
            allow_experimental: Allow experimental features
            overrides: Config overrides (merged into config)
            idempotency_key: Optional idempotency key for job submission

        Returns:
            RLJob instance

        Example:
            >>> job = RLJob.from_config(
            ...     config_path="configs/rl_gspo.toml",
            ...     synth_base_url="https://api.usesynth.ai",
            ...     synth_user_key=os.environ["SYNTH_API_KEY"],
            ...     localapi_url="https://my-task-app.usesynth.ai",
            ... )
        """
        config_path_obj = Path(config_path)

        # Resolve API key
        if not synth_user_key:
            synth_user_key = os.environ.get("SYNTH_API_KEY")
            if not synth_user_key:
                raise ValueError(
                    "synth_user_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        # Resolve LocalAPI URL
        if not localapi_url:
            localapi_url = os.environ.get("SYNTH_LOCALAPI_URL")

        # Resolve LocalAPI API key
        if not localapi_key:
            localapi_key = ensure_localapi_auth(
                synth_user_key=synth_user_key,
                synth_base_url=synth_base_url,
            )

        config = RLJobConfig(
            config_path=config_path_obj,
            synth_base_url=synth_base_url,
            synth_user_key=synth_user_key,
            localapi_url=localapi_url,
            localapi_key=localapi_key,
            allow_experimental=allow_experimental,
            overrides=overrides,
            idempotency_key=idempotency_key,
        )

        return cls(config)

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        synth_user_key: Optional[str] = None,
        synth_base_url: Optional[str] = None,
    ) -> "RLJob":
        """Resume an existing RL job by ID.

        Args:
            job_id: Existing job ID
            synth_base_url: Backend API URL (defaults to SYNTH_BACKEND_URL)
            synth_user_key: API key (defaults to env var SYNTH_API_KEY)

        Returns:
            RLJob instance for the existing job

        Example:
            >>> job = RLJob.from_job_id(
            ...     job_id="rl_abc123",
            ...     synth_base_url="https://api.usesynth.ai",
            ...     synth_user_key=os.environ["SYNTH_API_KEY"],
            ... )
        """
        # Resolve API key
        if not synth_user_key:
            synth_user_key = os.environ.get("SYNTH_API_KEY")
            if not synth_user_key:
                raise ValueError(
                    "synth_user_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        # Create minimal config (we don't need the config file for resuming)
        config = RLJobConfig(
            config_path=Path("/dev/null"),  # Dummy path
            synth_base_url=synth_base_url,
            synth_user_key=synth_user_key,
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
            overrides["synth_base_url"] = self.config.synth_base_url
            if self.config.localapi_url:
                overrides["localapi_url"] = self.config.localapi_url

            self._build_result = build_rl_payload(
                config_path=self.config.config_path,
                localapi_url=self.config.localapi_url or "",
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
            ValueError: If LocalAPI health check fails
        """
        ctx: Dict[str, Any] = {"config_path": str(self.config.config_path)}
        log_info("RLJob.submit invoked", ctx=ctx)
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check:
            localapi_key = self.config.localapi_key or ""
            health = check_local_api_health(build.localapi_url, localapi_key)
            if not health.ok:
                raise ValueError(f"Task app health check failed: {health.detail}")

        # Submit job
        create_url = synth_rl_jobs_url(self.config.synth_base_url)
        headers = {
            "Authorization": f"Bearer {self.config.synth_user_key}",
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
                    f"\n3. Verify backend is running at: {self.config.synth_base_url}"
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
                synth_user_key=self.config.synth_user_key,
                synth_base_url=self.config.synth_base_url,
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
            base_url=synth_api_base(self.config.synth_base_url),
            synth_user_key=self.config.synth_user_key,
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
