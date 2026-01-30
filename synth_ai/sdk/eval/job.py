"""First-class SDK API for evaluation jobs.

This module provides high-level abstractions for running evaluation jobs
that route through the backend for trace capture and cost tracking.

Example:
    from synth_ai.sdk.eval import EvalJob, EvalResult

    job = EvalJob(config)
    job.submit()

    # progress=True provides built-in status printing:
    # [00:05] running | 3/10 completed
    # [00:10] running | 7/10 completed
    # [00:15] completed | mean_reward: 0.85
    result = job.poll_until_complete(progress=True)

    # Typed result access (not raw dict)
    if result.succeeded:
        print(f"Mean reward: {result.mean_reward}")
        print(f"Total cost: ${result.total_cost_usd:.4f}")
        for seed_result in result.seed_results:
            print(f"  Seed {seed_result['seed']}: {seed_result['score']}")
    elif result.failed:
        print(f"Error: {result.error}")

See Also:
    - `synth_ai.cli.eval`: CLI implementation
    - `synth_ai.sdk.optimization`: Similar pattern for optimization jobs
"""

import contextlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.eval.") from exc

from synth_ai.core.utils.urls import BACKEND_URL_BASE, is_synthtunnel_url
from synth_ai.sdk.localapi.auth import ensure_localapi_auth


def _require_rust() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for eval jobs. Install rust bindings.")
    return synth_ai_py


def _infer_provider(model: str | None) -> str | None:
    if not model:
        return None
    lowered = model.lower()
    if lowered.startswith("gpt") or "openai" in lowered:
        return "openai"
    if lowered.startswith("claude") or "anthropic" in lowered:
        return "anthropic"
    return None


class EvalStatus(str, Enum):
    """Status of an evaluation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> "EvalStatus":
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
        ...     print(f"Mean reward: {result.mean_reward:.2%}")
        ...     print(f"Total cost: ${result.total_cost_usd:.4f}")
        >>> else:
        ...     print(f"Failed: {result.error}")
    """

    job_id: str
    status: EvalStatus
    mean_reward: Optional[float] = None
    total_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    num_completed: int = 0
    num_total: int = 0
    seed_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> "EvalResult":
        """Create result from API response dict."""
        status_str = data.get("status", "pending")
        status = EvalStatus.from_string(status_str)

        # Extract summary metrics
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        results_info = data.get("results", {})

        # Handle both summary dict and inline fields
        mean_reward = summary.get("mean_reward") or data.get("mean_reward")
        if mean_reward is None and isinstance(results_info, dict):
            mean_reward = results_info.get("mean_reward")
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
            mean_reward=mean_reward,
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
        task_app_worker_token: SynthTunnel worker token for relay auth (required for st.usesynth.ai URLs).
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
    task_app_worker_token: Optional[str] = field(default=None, repr=False)
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
        if not self.env_name:
            self.env_name = "default"

        if is_synthtunnel_url(self.task_app_url):
            if not (self.task_app_worker_token or "").strip():
                raise ValueError(
                    "task_app_worker_token is required for SynthTunnel task_app_url. "
                    "Pass tunnel.worker_token when submitting jobs."
                )
            # Do not send ENVIRONMENT_API_KEY to SynthTunnel gateway
            self.task_app_api_key = None
        else:
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
        >>> print(f"Mean reward: {results['summary']['mean_reward']}")
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
        self._rust = _require_rust()
        self._client = self._rust.SynthClient(self.config.api_key, self.config.backend_url)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        task_app_worker_token: Optional[str] = None,
        task_app_url: Optional[str] = None,
        seeds: Optional[List[int]] = None,
    ) -> "EvalJob":
        """Create a job from a TOML config file.

        Loads evaluation configuration from a TOML file and allows
        overriding specific values via arguments.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            task_app_api_key: Task app API key (defaults to ENVIRONMENT_API_KEY)
            task_app_worker_token: SynthTunnel worker token for relay auth
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
        try:
            import synth_ai_py
        except Exception as exc:  # pragma: no cover - rust bindings required
            raise RuntimeError("synth_ai_py is required for eval config parsing.") from exc

        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        toml_data = synth_ai_py.load_toml(str(config_path_obj))
        if not isinstance(toml_data, dict):
            toml_data = dict(toml_data)

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
                    "verifier_config": pl_config.get("verifier", {})
                    if isinstance(pl_config.get("verifier"), dict)
                    else None,
                }

        # Resolve API key
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        # Build config with overrides
        final_task_app_url = (
            task_app_url or eval_config.get("url") or eval_config.get("task_app_url")
        )
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
            task_app_worker_token=task_app_worker_token,
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
    ) -> "EvalJob":
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
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

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
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        # Build job request payload
        policy = dict(self.config.policy_config)
        if "provider" not in policy:
            inferred = _infer_provider(policy.get("model"))
            if inferred:
                policy["provider"] = inferred

        job_request = {
            "task_app_url": self.config.task_app_url,
            "task_app_api_key": self.config.task_app_api_key,
            "task_app_worker_token": self.config.task_app_worker_token,
            "app_id": self.config.app_id,
            "env_name": self.config.env_name,
            "seeds": self.config.seeds,
            "policy": policy,
            "env_config": self.config.env_config,
            "verifier_config": self.config.verifier_config,
            "max_concurrent": self.config.concurrency,
            "timeout": self.config.timeout,
        }

        job_id = self._client.submit_eval(job_request)
        self._job_id = job_id
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
            ...     print(f"Mean reward: {status['results']['mean_reward']}")
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return self._client.get_eval_status(self._job_id)

    def poll_until_complete(
        self,
        *,
        timeout: float = 1200.0,
        interval: float = 15.0,
        progress: bool = False,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> EvalResult:
        """Poll job until it reaches a terminal state, then return results.

        Polls the backend until the job completes or fails, then fetches
        and returns the detailed results.

        Args:
            timeout: Maximum seconds to wait (default: 1200 = 20 minutes)
            interval: Seconds between poll attempts (default: 15)
            progress: If True, print status updates during polling (useful for notebooks)
            on_status: Optional callback called on each status update (for custom progress handling)

        Returns:
            EvalResult with typed status, mean_reward, seed_results, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout is exceeded

        Example:
            >>> result = job.poll_until_complete(progress=True)
            [00:05] running | 3/10 completed
            [00:10] running | 7/10 completed
            [00:15] completed | mean_reward: 0.85
            >>> result.succeeded
            True
            >>> result.mean_reward
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
                completed = (
                    results_info.get("completed", 0) if isinstance(results_info, dict) else 0
                )
                total = (
                    results_info.get("total", len(self.config.seeds))
                    if isinstance(results_info, dict)
                    else len(self.config.seeds)
                )

                # Progress output
                if progress:
                    mins, secs = divmod(int(elapsed), 60)
                    if status.is_terminal:
                        # Get final results for mean_reward
                        try:
                            final_results = self.get_results()
                            mean_reward = final_results.get("summary", {}).get("mean_reward")
                            if mean_reward is None:
                                results_info = final_results.get("results", {})
                                if isinstance(results_info, dict):
                                    mean_reward = results_info.get("mean_reward")
                            reward_str = (
                                f"mean_reward: {mean_reward:.2f}" if mean_reward is not None else ""
                            )
                            print(f"[{mins:02d}:{secs:02d}] {status.value} | {reward_str}")
                            # Use final results for the return value
                            last_data = final_results
                        except Exception:
                            print(f"[{mins:02d}:{secs:02d}] {status.value}")
                    else:
                        print(
                            f"[{mins:02d}:{secs:02d}] {status.value} | {completed}/{total} completed"
                        )

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

            time.sleep(interval)

    def stream_until_complete(
        self,
        *,
        timeout: float = 1200.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> EvalResult:
        """Stream job events until completion using SSE.

        This provides real-time event streaming instead of polling,
        reducing server load and providing faster updates.

        Args:
            timeout: Maximum seconds to wait (default: 1200 = 20 minutes)
            interval: Seconds between status checks (for SSE reconnects)
            handlers: Optional StreamHandler instances for custom event handling
            on_event: Optional callback called on each event

        Returns:
            EvalResult with typed status, mean_reward, seed_results, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout exceeded

        Example:
            >>> result = job.stream_until_complete()
            [00:05] Eval started: 10 seeds
            [00:10] Progress: 5/10 seeds completed
            [00:15] Eval completed: mean_reward=0.85
            >>> result.succeeded
            True
        """
        import asyncio
        import contextlib

        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        from synth_ai.core.streaming import (
            EvalHandler,
            JobStreamer,
            StreamConfig,
            StreamEndpoints,
            StreamType,
        )

        # Build stream config
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS},
            max_events_per_poll=500,
            deduplicate=True,
        )

        # Use provided handlers or default EvalHandler
        if handlers is None:
            handlers = [EvalHandler()]

        # Create streamer with eval endpoints
        # Note: base_url should NOT include /api prefix - JobStreamer adds it
        base_url = self._base_url().replace("/api", "").rstrip("/")
        streamer = JobStreamer(
            base_url=base_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            endpoints=StreamEndpoints.eval(self._job_id),
            config=config,
            handlers=list(handlers),
            interval_seconds=interval,
            timeout_seconds=timeout,
        )

        # Run streaming
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in an async context - create a new thread to run the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, streamer.stream_until_terminal())
                final_status = future.result()
        else:
            final_status = asyncio.run(streamer.stream_until_terminal())

        # Callback for custom handling
        if on_event and isinstance(final_status, dict):
            with contextlib.suppress(Exception):
                on_event(final_status)

        # Fetch full results if completed
        status_str = str(final_status.get("status", "")).lower()
        if status_str == "completed":
            try:
                full_results = self.get_results()
                return EvalResult.from_response(self._job_id, full_results)
            except Exception:
                pass

        return EvalResult.from_response(self._job_id, final_status)

    async def stream_sse_until_complete_async(
        self,
        *,
        timeout: float = 1200.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        progress: bool = True,
    ) -> EvalResult:
        """Stream job events via SSE until completion (async version).

        This provides real-time event streaming instead of polling,
        reducing latency and providing instant updates.

        Args:
            timeout: Maximum seconds to wait (default: 1200 = 20 minutes)
            on_event: Optional callback called on each event
            progress: If True, print progress updates

        Returns:
            EvalResult with typed status, mean_reward, seed_results, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        from synth_ai.core.rust_core.sse import stream_sse_events

        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        job_id = self._job_id
        base_url = self._base_url()
        sse_url = f"{base_url}/eval/jobs/{job_id}/events/stream"

        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        start_time = time.time()
        completed = 0
        total = len(self.config.seeds)
        last_status = "pending"
        terminal_events = {
            "eval.policy.job.completed",
            "eval.policy.job.failed",
            "job.completed",
            "job.failed",
        }

        if progress:
            print(f"[DEBUG] SSE stream connecting to {sse_url}")

        try:
            async for event in stream_sse_events(
                sse_url,
                headers=headers,
                timeout=timeout,
            ):
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    if progress:
                        print(f"[stream] timeout after {timeout:.0f}s")
                    break

                # Call event handler
                if on_event:
                    with contextlib.suppress(Exception):
                        on_event(event)

                # Extract progress info
                event_type = event.get("type", "")
                event_data = event.get("data", {})

                # Track completion progress
                if "completed" in event_data:
                    completed = event_data.get("completed", completed)
                if "total" in event_data:
                    total = event_data.get("total", total)
                if event_type in ("eval.policy.seed.completed", "seed.completed"):
                    completed += 1

                # Progress output
                if progress:
                    mins, secs = divmod(int(elapsed), 60)
                    msg = event.get("message", event_type)
                    print(f"[{mins:02d}:{secs:02d}] {event_type}: {msg} | {completed}/{total}")

                # Check for terminal event
                if event_type in terminal_events:
                    last_status = "completed" if "completed" in event_type else "failed"
                    break

        except Exception as exc:
            if progress:
                print(f"[stream] SSE error: {exc}, falling back to polling")
            return self.poll_until_complete(timeout=timeout, progress=progress, on_event=on_event)

        # Fetch full results
        try:
            final_results = self.get_results()
            return EvalResult.from_response(job_id, final_results)
        except Exception:
            return EvalResult.from_response(job_id, {"status": last_status, "job_id": job_id})

    def stream_sse_until_complete(
        self,
        *,
        timeout: float = 1200.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        progress: bool = True,
    ) -> EvalResult:
        """Stream job events via SSE until completion (sync wrapper).

        This provides real-time event streaming instead of polling.

        Args:
            timeout: Maximum seconds to wait (default: 1200 = 20 minutes)
            on_event: Optional callback called on each event
            progress: If True, print progress updates

        Returns:
            EvalResult with typed status, mean_reward, seed_results, etc.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in an async context - create a new thread to run the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.stream_sse_until_complete_async(
                        timeout=timeout,
                        on_event=on_event,
                        progress=progress,
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.stream_sse_until_complete_async(
                    timeout=timeout,
                    on_event=on_event,
                    progress=progress,
                )
            )

    def get_results(self) -> Dict[str, Any]:
        """Get detailed job results.

        Fetches the full results including per-seed scores, tokens, and costs.

        Returns:
            Results dictionary with:
            - job_id: Job identifier
            - status: Job status
            - summary: Aggregate metrics
                - mean_reward: Average reward across seeds
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

        return self._client.get_eval_results(self._job_id)

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

        output_path = Path(output_dir)

        payload = self._client.download_eval_traces(self._job_id)
        output_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(output_path)
        return output_path

    def cancel(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a running eval job.

        Sends a cancellation request to the backend. The job will stop
        at the next checkpoint and emit a cancelled status event.

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
            RuntimeError: If the cancellation request fails

        Example:
            >>> job.submit()
            >>> # Later...
            >>> result = job.cancel(reason="No longer needed")
            >>> print(result["message"])
            "Temporal workflow cancelled successfully."
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return self._client.cancel_eval(self._job_id, reason)

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

        return self._client.query_eval_workflow_state(self._job_id)


__all__ = ["EvalJob", "EvalJobConfig", "EvalResult", "EvalStatus"]
