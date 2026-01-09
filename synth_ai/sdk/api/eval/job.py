"""First-class SDK API for evaluation jobs.

This module provides high-level abstractions for running evaluation jobs
that route through the backend for trace capture and cost tracking.

Example:
    from synth_ai.sdk.api.eval import EvalJob, EvalResult

    job = EvalJob(config)
    job.submit()

    # progress=True provides built-in status printing:
    # [00:05] running | 3/10 completed
    # [00:10] running | 7/10 completed
    # [00:15] completed | mean_score: 0.85
    result = job.poll_until_complete(progress=True)

    # Typed result access (not raw dict)
    if result.succeeded:
        print(f"Mean score: {result.mean_score}")
        print(f"Total cost: ${result.total_cost_usd:.4f}")
        for seed_result in result.seed_results:
            print(f"  Seed {seed_result['seed']}: {seed_result['score']}")
    elif result.failed:
        print(f"Error: {result.error}")

See Also:
    - `synth_ai.cli.commands.eval`: CLI implementation3
    - `synth_ai.sdk.api.train.prompt_learning`: Similar pattern for training
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from synth_ai.core.telemetry import log_info
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.localapi.auth import ensure_localapi_auth


class EvalStatus(str, Enum):
    """Status of an evaluation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> EvalStatus:
        """Convert string to EvalStatus, defaulting to PENDING for unknown values."""
        try:
            return cls(status.lower())
        except ValueError:
            return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        """Whether this status is terminal (job won't change further)."""
        return self in (EvalStatus.COMPLETED, EvalStatus.FAILED, EvalStatus.CANCELLED)

    @property
    def is_success(self) -> bool:
        """Whether this status indicates success."""
        return self == EvalStatus.COMPLETED


@dataclass
class EvalResult:
    """Typed result from an evaluation job.

    Provides clean accessors for common fields instead of raw dict access.

    Example:
        >>> result = job.poll_until_complete(progress=True)
        >>> if result.succeeded:
        ...     print(f"Mean score: {result.mean_score:.2%}")
        ...     print(f"Total cost: ${result.total_cost_usd:.4f}")
        >>> else:
        ...     print(f"Failed: {result.error}")
    """

    job_id: str
    status: EvalStatus
    mean_score: Optional[float] = None
    total_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    num_completed: int = 0
    num_total: int = 0
    seed_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> EvalResult:
        """Create result from API response dict."""
        status_str = data.get("status", "pending")
        status = EvalStatus.from_string(status_str)

        # Extract summary metrics
        summary = data.get("summary", {})
        results_info = data.get("results", {})

        # Handle both summary dict and inline fields
        mean_score = summary.get("mean_score") or data.get("mean_score")
        total_tokens = summary.get("total_tokens") or data.get("total_tokens")
        total_cost_usd = summary.get("total_cost_usd") or data.get("total_cost_usd")

        # Get completion progress
        num_completed = results_info.get("completed", 0) if isinstance(results_info, dict) else 0
        num_total = results_info.get("total", 0) if isinstance(results_info, dict) else 0

        # Get per-seed results (can be in "results" list or nested)
        seed_results = data.get("results", [])
        if isinstance(seed_results, dict):
            seed_results = seed_results.get("items", [])

        return cls(
            job_id=job_id,
            status=status,
            mean_score=mean_score,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            num_completed=num_completed,
            num_total=num_total,
            seed_results=list(seed_results) if isinstance(seed_results, list) else [],
            error=data.get("error"),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        """Whether the job completed successfully."""
        return self.status.is_success

    @property
    def failed(self) -> bool:
        """Whether the job failed."""
        return self.status == EvalStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.status.is_terminal


@dataclass
class EvalJobConfig:
    """Configuration for an evaluation job.

    This dataclass holds all the configuration needed to submit and run
    an evaluation job via the backend.

    Attributes:
        task_app_url: URL of the task app to evaluate (e.g., "http://localhost:8103").
            Required for job submission. Alias: local_api_url
        backend_url: Base URL of the Synth API backend (e.g., "https://api.usesynth.ai").
            Can also be set via SYNTH_BASE_URL or BACKEND_BASE_URL environment variables.
        api_key: Synth API key for authentication with the backend.
            Can also be set via SYNTH_API_KEY environment variable.
        task_app_api_key: API key for authenticating with the task app.
            Defaults to ENVIRONMENT_API_KEY env var if not provided. Alias: local_api_key
        app_id: Task app identifier (optional, for logging/tracking).
        env_name: Environment name within the task app.
        seeds: List of seeds/indices to evaluate.
        policy_config: Model and provider configuration for the policy.
        env_config: Additional environment configuration.
        concurrency: Maximum number of parallel rollouts (default: 5).
        timeout: Maximum seconds per rollout (default: 600.0).

    Example:
        >>> config = EvalJobConfig(
        ...     task_app_url="http://localhost:8103",
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ...     env_name="banking77",
        ...     seeds=[0, 1, 2, 3, 4],
        ...     policy_config={"model": "gpt-4", "provider": "openai"},
        ... )
    """

    task_app_url: str = field(default="")
    api_key: str = field(default="")
    backend_url: Optional[str] = field(default="")
    task_app_api_key: Optional[str] = None
    app_id: Optional[str] = None
    env_name: Optional[str] = None
    seeds: List[int] = field(default_factory=list)
    policy_config: Dict[str, Any] = field(default_factory=dict)
    env_config: Dict[str, Any] = field(default_factory=dict)
    verifier_config: Optional[Dict[str, Any]] = None
    concurrency: int = 5
    timeout: float = 600.0
    # Aliases for backwards compatibility (not stored, just used in __init__)
    local_api_url: str = field(default="", repr=False)
    local_api_key: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and handle aliases."""
        # Handle aliases for backwards compatibility
        if self.local_api_url and not self.task_app_url:
            self.task_app_url = self.local_api_url
        if self.local_api_key and not self.task_app_api_key:
            self.task_app_api_key = self.local_api_key

        if not self.task_app_url:
            raise ValueError("task_app_url (or local_api_url) is required")
        # Use backend_url from config if provided, otherwise fall back to BACKEND_URL_BASE
        if not self.backend_url:
            self.backend_url = BACKEND_URL_BASE
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.seeds:
            raise ValueError("seeds list is required and cannot be empty")

        # Get task_app_api_key from environment if not provided
        if not self.task_app_api_key:
            self.task_app_api_key = ensure_localapi_auth(
                backend_base=self.backend_url,
                synth_api_key=self.api_key,
            )


class EvalJob:
    """High-level SDK class for running evaluation jobs via the backend.

    This class provides a clean API for:
    1. Submitting evaluation jobs to the backend
    2. Polling job status until completion
    3. Retrieving detailed results with metrics, tokens, and costs
    4. Downloading traces for analysis

    The backend routes LLM calls through the inference interceptor, which:
    - Captures traces automatically
    - Tracks token usage
    - Calculates costs based on model pricing

    Example:
        >>> from synth_ai.sdk.api.eval import EvalJob
        >>>
        >>> # Create job from config file
        >>> job = EvalJob.from_config(
        ...     config_path="banking77_eval.toml",
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key=os.environ["SYNTH_API_KEY"],
        ... )
        >>>
        >>> # Submit job
        >>> job_id = job.submit()
        >>> print(f"Job submitted: {job_id}")
        >>>
        >>> # Poll until complete
        >>> results = job.poll_until_complete(timeout=1200.0)
        >>> print(f"Mean score: {results['summary']['mean_score']}")
        >>>
        >>> # Download traces
        >>> job.download_traces("./traces")

    See Also:
        - `PromptLearningJob`: Similar pattern for prompt learning jobs
        - Backend API: POST /api/eval/jobs, GET /api/eval/jobs/{job_id}
    """

    # Default poll settings
    _POLL_INTERVAL_S = 2.0
    _MAX_POLL_ATTEMPTS = 600  # 20 minutes max

    def __init__(
        self,
        config: EvalJobConfig,
        job_id: Optional[str] = None,
    ) -> None:
        """Initialize an evaluation job.

        Args:
            config: Job configuration with task app URL, seeds, policy, etc.
            job_id: Existing job ID (if resuming a previous job)
        """
        self.config = config
        self._job_id = job_id

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        task_app_url: Optional[str] = None,
        seeds: Optional[List[int]] = None,
    ) -> EvalJob:
        """Create a job from a TOML config file.

        Loads evaluation configuration from a TOML file and allows
        overriding specific values via arguments.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            task_app_api_key: Task app API key (defaults to ENVIRONMENT_API_KEY)
            task_app_url: Override task app URL from config
            seeds: Override seeds list from config

        Returns:
            EvalJob instance ready for submission

        Raises:
            ValueError: If required config is missing
            FileNotFoundError: If config file doesn't exist

        Example:
            >>> job = EvalJob.from_config(
            ...     "banking77_eval.toml",
            ...     backend_url="https://api.usesynth.ai",
            ...     api_key="sk_live_...",
            ...     seeds=[0, 1, 2],  # Override seeds
            ... )
        """
        import tomllib

        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path_obj, "rb") as f:
            toml_data = tomllib.load(f)

        # Extract eval section (supports both [eval] and [prompt_learning] formats)
        eval_config = toml_data.get("eval", {})
        if not eval_config:
            pl_config = toml_data.get("prompt_learning", {})
            if pl_config:
                eval_config = {
                    "app_id": pl_config.get("task_app_id"),
                    "url": pl_config.get("task_app_url"),
                    "env_name": pl_config.get("gepa", {}).get("env_name"),
                    "seeds": pl_config.get("gepa", {}).get("evaluation", {}).get("seeds", []),
                    "policy_config": pl_config.get("gepa", {}).get("policy", {}),
                    "verifier_config": pl_config.get("verifier", {}) if isinstance(pl_config.get("verifier"), dict) else None,
                }

        # Resolve API key
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY env var)")

        # Build config with overrides
        final_task_app_url = task_app_url or eval_config.get("url") or eval_config.get("task_app_url")
        if not final_task_app_url:
            raise ValueError("task_app_url is required (in config or as argument)")

        final_seeds = seeds or eval_config.get("seeds", [])
        if not final_seeds:
            raise ValueError("seeds list is required (in config or as argument)")

        config = EvalJobConfig(
            task_app_url=final_task_app_url,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            app_id=eval_config.get("app_id"),
            env_name=eval_config.get("env_name"),
            seeds=list(final_seeds),
            policy_config=eval_config.get("policy_config", {}),
            env_config=eval_config.get("env_config", {}),
            verifier_config=eval_config.get("verifier_config"),
            concurrency=eval_config.get("concurrency", 5),
            timeout=eval_config.get("timeout", 600.0),
        )

        return cls(config)

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> EvalJob:
        """Resume an existing job by ID.

        Use this to check status or get results of a previously submitted job.

        Args:
            job_id: Existing job ID (e.g., "eval-abc123")
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            EvalJob instance for the existing job

        Example:
            >>> job = EvalJob.from_job_id("eval-abc123")
            >>> status = job.get_status()
            >>> if status["status"] == "completed":
            ...     results = job.get_results()
        """
        # Resolve API key
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY env var)")

        # Create minimal config for resumed job
        config = EvalJobConfig(
            task_app_url="resumed",  # Placeholder - not needed for status/results
            backend_url=backend_url,
            api_key=api_key,
            seeds=[0],  # Placeholder
        )

        return cls(config, job_id=job_id)

    def _base_url(self) -> str:
        """Get normalized base URL for API calls."""
        base = (self.config.backend_url or BACKEND_URL_BASE).rstrip("/")
        if not base.endswith("/api"):
            base = f"{base}/api"
        return base

    def _headers(self) -> Dict[str, str]:
        """Get headers for API calls."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def submit(self) -> str:
        """Submit the job to the backend.

        Creates an eval job on the backend which will:
        1. Route LLM calls through the inference interceptor
        2. Capture traces and token usage
        3. Calculate costs based on model pricing

        Returns:
            Job ID (e.g., "eval-abc123")

        Raises:
            RuntimeError: If job submission fails or job already submitted
            ValueError: If configuration is invalid

        Example:
            >>> job = EvalJob.from_config("eval.toml")
            >>> job_id = job.submit()
            >>> print(f"Submitted: {job_id}")
        """
        ctx: Dict[str, Any] = {"task_app_url": self.config.task_app_url}
        log_info("EvalJob.submit invoked", ctx=ctx)

        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        # Build job request payload
        policy = dict(self.config.policy_config)

        job_request = {
            "task_app_url": self.config.task_app_url,
            "task_app_api_key": self.config.task_app_api_key,
            "app_id": self.config.app_id,
            "env_name": self.config.env_name,
            "seeds": self.config.seeds,
            "policy": policy,
            "env_config": self.config.env_config,
            "verifier_config": self.config.verifier_config,
            "max_concurrent": self.config.concurrency,
            "timeout": self.config.timeout,
        }

        # Submit synchronously using httpx
        url = f"{self._base_url()}/eval/jobs"

        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            resp = client.post(url, json=job_request, headers=self._headers())

            if resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
                )

            job_data = resp.json()
            job_id = job_data.get("job_id")
            if not job_id:
                raise RuntimeError(f"No job_id in response: {job_data}")

            self._job_id = job_id
            ctx["job_id"] = job_id
            log_info("EvalJob.submit completed", ctx=ctx)
            return job_id

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        return self._job_id

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Job status dictionary with keys:
            - job_id: Job identifier
            - status: "running", "completed", or "failed"
            - error: Error message if failed
            - created_at, started_at, completed_at: Timestamps
            - config: Original job configuration
            - results: Summary results if completed

        Raises:
            RuntimeError: If job hasn't been submitted yet

        Example:
            >>> status = job.get_status()
            >>> print(f"Status: {status['status']}")
            >>> if status["status"] == "completed":
            ...     print(f"Mean score: {status['results']['mean_score']}")
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self._base_url()}/eval/jobs/{self._job_id}"

        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            resp = client.get(url, headers=self._headers())

            if resp.status_code != 200:
                raise RuntimeError(f"Failed to get status: {resp.status_code} {resp.text}")

            return resp.json()

    def poll_until_complete(
        self,
        *,
        timeout: float = 1200.0,
        interval: float = 2.0,
        progress: bool = False,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> EvalResult:
        """Poll job until it reaches a terminal state, then return results.

        Polls the backend until the job completes or fails, then fetches
        and returns the detailed results.

        Args:
            timeout: Maximum seconds to wait (default: 1200 = 20 minutes)
            interval: Seconds between poll attempts (default: 2)
            progress: If True, print status updates during polling (useful for notebooks)
            on_status: Optional callback called on each status update (for custom progress handling)

        Returns:
            EvalResult with typed status, mean_score, seed_results, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout is exceeded

        Example:
            >>> result = job.poll_until_complete(progress=True)
            [00:05] running | 3/10 completed
            [00:10] running | 7/10 completed
            [00:15] completed | mean_score: 0.85
            >>> result.succeeded
            True
            >>> result.mean_score
            0.85
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        job_id = self._job_id
        start_time = time.time()
        last_data: Dict[str, Any] = {}

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                if progress:
                    print(f"[poll] timeout after {timeout:.0f}s")
                # Return with whatever data we have
                return EvalResult.from_response(job_id, last_data)

            try:
                status_data = self.get_status()
                last_data = status_data

                status = EvalStatus.from_string(status_data.get("status", "pending"))

                # Extract progress info
                results_info = status_data.get("results", {})
                completed = results_info.get("completed", 0) if isinstance(results_info, dict) else 0
                total = results_info.get("total", len(self.config.seeds)) if isinstance(results_info, dict) else len(self.config.seeds)

                # Progress output
                if progress:
                    mins, secs = divmod(int(elapsed), 60)
                    if status.is_terminal:
                        # Get final results for mean_score
                        try:
                            final_results = self.get_results()
                            mean_score = final_results.get("summary", {}).get("mean_score")
                            score_str = f"mean_score: {mean_score:.2f}" if mean_score is not None else ""
                            print(f"[{mins:02d}:{secs:02d}] {status.value} | {score_str}")
                            # Use final results for the return value
                            last_data = final_results
                        except Exception:
                            print(f"[{mins:02d}:{secs:02d}] {status.value}")
                    else:
                        print(f"[{mins:02d}:{secs:02d}] {status.value} | {completed}/{total} completed")

                # Callback for custom handling
                if on_status:
                    on_status(status_data)

                # Check terminal state
                if status.is_terminal:
                    # Fetch full results if completed
                    if status == EvalStatus.COMPLETED:
                        try:
                            final_results = self.get_results()
                            return EvalResult.from_response(job_id, final_results)
                        except Exception:
                            pass
                    return EvalResult.from_response(job_id, last_data)

            except Exception as exc:
                if progress:
                    print(f"[poll] error: {exc}")
                log_info("poll request failed", ctx={"error": str(exc), "job_id": job_id})

            time.sleep(interval)

    def get_results(self) -> Dict[str, Any]:
        """Get detailed job results.

        Fetches the full results including per-seed scores, tokens, and costs.

        Returns:
            Results dictionary with:
            - job_id: Job identifier
            - status: Job status
            - summary: Aggregate metrics
                - mean_score: Average score across seeds
                - total_tokens: Total token usage
                - total_cost_usd: Total cost
                - num_seeds: Number of seeds evaluated
                - num_successful: Seeds that completed
                - num_failed: Seeds that failed
            - results: List of per-seed results
                - seed: Seed number
                - score: Evaluation score
                - tokens: Token count
                - cost_usd: Cost for this seed
                - latency_ms: Execution time
                - error: Error message if failed

        Raises:
            RuntimeError: If job hasn't been submitted yet

        Example:
            >>> results = job.get_results()
            >>> for r in results["results"]:
            ...     print(f"Seed {r['seed']}: score={r['score']}, tokens={r['tokens']}")
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self._base_url()}/eval/jobs/{self._job_id}/results"

        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            resp = client.get(url, headers=self._headers())

            if resp.status_code != 200:
                raise RuntimeError(f"Failed to get results: {resp.status_code} {resp.text}")

            return resp.json()

    def download_traces(self, output_dir: str | Path) -> Path:
        """Download traces for the job to a directory.

        Downloads the traces ZIP file from the backend and extracts
        it to the specified directory.

        Args:
            output_dir: Directory to extract traces to

        Returns:
            Path to the output directory

        Raises:
            RuntimeError: If job hasn't been submitted or download fails

        Example:
            >>> traces_dir = job.download_traces("./traces")
            >>> for trace_file in traces_dir.glob("*.json"):
            ...     print(f"Trace: {trace_file}")
        """
        import io
        import zipfile

        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self._base_url()}/eval/jobs/{self._job_id}/traces"
        output_path = Path(output_dir)

        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            resp = client.get(url, headers=self._headers())

            if resp.status_code != 200:
                raise RuntimeError(f"Failed to download traces: {resp.status_code} {resp.text}")

            output_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(output_path)

            return output_path


__all__ = ["EvalJob", "EvalJobConfig", "EvalResult", "EvalStatus"]
