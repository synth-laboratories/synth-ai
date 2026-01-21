"""Graph optimization job implementation.

This module provides the canonical `GraphOptimizationJob` class for running
graph/workflow optimization jobs.

Replaces: `GraphOptimizationJob` (from sdk/api/train), `GraphEvolveJob` (deprecated)
Backend endpoint: `/api/graph-evolve/jobs`, `/api/graph_evolve/jobs`

Algorithms:
- graph_gepa: Graph Genetic Evolutionary Prompt Algorithm (default, currently only)

Two construction modes:
- `from_config()`: TOML config file (like old GraphOptimizationJob)
- `from_dataset()`: JSON dataset (like GraphEvolveJob)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence

from synth_ai.core.utils.urls import BACKEND_URL_BASE

if TYPE_CHECKING:
    from synth_ai.core.streaming import StreamHandler


class Algorithm(str, Enum):
    """Supported graph optimization algorithms."""

    GRAPH_GEPA = "graph_gepa"

    @classmethod
    def from_string(cls, value: str) -> Algorithm:
        """Convert string to Algorithm enum."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.GRAPH_GEPA


class JobStatus(str, Enum):
    """Status of a graph optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCEEDED = "succeeded"  # Alias
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
        return self in (
            JobStatus.COMPLETED,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        """Whether this status indicates success."""
        return self in (JobStatus.COMPLETED, JobStatus.SUCCEEDED)


@dataclass
class GraphOptimizationResult:
    """Typed result from a graph optimization job.

    Provides clean accessors for common fields instead of raw dict access.
    """

    job_id: str
    status: JobStatus
    algorithm: Algorithm = Algorithm.GRAPH_GEPA
    best_score: Optional[float] = None
    best_yaml: Optional[str] = None
    best_snapshot_id: Optional[str] = None
    generations_completed: Optional[int] = None
    total_candidates_evaluated: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> GraphOptimizationResult:
        """Create result from API response dict."""
        status_str = data.get("status", "pending")
        status = JobStatus.from_string(status_str)
        best_score = data.get("best_score") or data.get("best_reward")

        return cls(
            job_id=job_id,
            status=status,
            algorithm=Algorithm.GRAPH_GEPA,
            best_score=best_score,
            best_yaml=data.get("best_yaml"),
            best_snapshot_id=data.get("best_snapshot_id"),
            generations_completed=data.get("generations_completed"),
            total_candidates_evaluated=data.get("total_candidates_evaluated"),
            duration_seconds=data.get("duration_seconds"),
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


@dataclass
class GraphOptimizationJobConfig:
    """Configuration for a graph optimization job.

    This dataclass holds all the configuration needed to submit and run
    a graph optimization job.

    Supports two modes:
    1. **Config-based**: Provide `config_path` pointing to a TOML file
    2. **Dataset-based**: Internal use via from_dataset() constructor

    Attributes:
        backend_url: Base URL of the Synth API backend.
        api_key: Synth API key for authentication.
        config_path: Path to the TOML configuration file (config mode).
        config_dict: Dictionary configuration (programmatic mode).
        algorithm: Optimization algorithm to use.
    """

    backend_url: str
    api_key: str
    config_path: Optional[Path] = None
    config_dict: Optional[Dict[str, Any]] = None
    algorithm: Algorithm = Algorithm.GRAPH_GEPA

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")


class GraphOptimizationJob:
    """High-level SDK class for running graph optimization jobs.

    This is the canonical class for graph optimization, unifying the old
    `GraphOptimizationJob` (config-based) and `GraphEvolveJob` (dataset-based)
    under a single API.

    Two construction modes:
    - `from_config()`: Create from TOML config file
    - `from_dataset()`: Create from JSON dataset (auto-generates task app)

    Example (config-based):
        >>> from synth_ai.sdk.optimization.graph import GraphOptimizationJob
        >>>
        >>> job = GraphOptimizationJob.from_config("my_config.toml")
        >>> job.submit()
        >>> result = job.stream_until_complete()
        >>> print(f"Best score: {result.best_score}")

    Example (dataset-based):
        >>> job = GraphOptimizationJob.from_dataset(
        ...     "tasks.json",
        ...     policy_models="gpt-4o-mini",
        ...     rollout_budget=100,
        ... )
        >>> job.submit()
        >>> result = job.stream_until_complete()
        >>> print(f"Best score: {result.best_score}")

    Attributes:
        job_id: The job ID (None until submitted)
        algorithm: The optimization algorithm being used
    """

    def __init__(
        self,
        config: GraphOptimizationJobConfig,
        job_id: Optional[str] = None,
        *,
        _delegate: Optional[Any] = None,
        _mode: str = "config",
    ) -> None:
        """Initialize a graph optimization job.

        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
            _delegate: Internal delegate object (config or dataset job)
            _mode: Internal mode indicator ("config" or "dataset")
        """
        self.config = config
        self._job_id = job_id
        self._delegate = _delegate
        self._mode = _mode
        self._algorithm = config.algorithm

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphOptimizationJob:
        """Create a job from a TOML config file.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            GraphOptimizationJob instance

        Raises:
            ValueError: If required config is missing
            FileNotFoundError: If config file doesn't exist
        """
        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = GraphOptimizationJobConfig(
            backend_url=backend_url,
            api_key=api_key,
            config_path=Path(config_path),
        )

        # Create delegate using old GraphOptimizationJob
        from synth_ai.sdk.optimization._impl.graph_optimization import (
            GraphOptimizationJob as LegacyGraphOptimizationJob,
        )

        delegate = LegacyGraphOptimizationJob.from_config(
            config_path=config_path,
            backend_url=backend_url,
            api_key=api_key,
        )

        return cls(config, _delegate=delegate, _mode="config")

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphOptimizationJob:
        """Create a job from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            GraphOptimizationJob instance
        """
        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = GraphOptimizationJobConfig(
            backend_url=backend_url,
            api_key=api_key,
            config_dict=config_dict,
        )

        # Create delegate using old GraphOptimizationJob
        from synth_ai.sdk.optimization._impl.graph_optimization import (
            GraphOptimizationJob as LegacyGraphOptimizationJob,
        )

        delegate = LegacyGraphOptimizationJob.from_dict(
            config_dict=config_dict,
            backend_url=backend_url,
            api_key=api_key,
        )

        return cls(config, _delegate=delegate, _mode="config")

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,  # str | Path | Dict[str, Any] | GraphEvolveTaskSet
        *,
        policy_models: str | List[str],
        rollout_budget: int = 100,
        proposer_effort: Literal["low", "medium", "high"] = "medium",
        judge_model: Optional[str] = None,
        judge_provider: Optional[str] = None,
        population_size: int = 4,
        num_generations: Optional[int] = None,
        problem_spec: Optional[str] = None,
        target_llm_calls: Optional[int] = None,
        graph_type: Optional[Literal["policy", "verifier", "rlm"]] = None,
        initial_graph_id: Optional[str] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_start: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphOptimizationJob:
        """Create a job from a JSON dataset (GraphEvolve style).

        This mode auto-generates task apps from the dataset - no user-managed
        task apps required.

        Args:
            dataset: Dataset as file path, dict, or GraphEvolveTaskSet object
            policy_models: Model(s) to use for policy inference
            rollout_budget: Total number of rollouts for optimization
            proposer_effort: Proposer effort level (low, medium, high)
            judge_model: Override judge model from dataset
            judge_provider: Override judge provider from dataset
            population_size: Population size for GEPA
            num_generations: Number of generations (auto-calculated if not specified)
            problem_spec: Detailed problem specification for the graph proposer
            target_llm_calls: Target number of LLM calls for the graph (1-10)
            graph_type: Type of graph to train ("policy", "verifier", or "rlm")
            initial_graph_id: Optional graph ID to warm-start optimization from
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            auto_start: Whether to start the job immediately
            metadata: Additional metadata for the job

        Returns:
            GraphOptimizationJob instance

        Example:
            >>> job = GraphOptimizationJob.from_dataset(
            ...     "tasks.json",
            ...     policy_models="gpt-4o-mini",
            ...     rollout_budget=100,
            ... )
            >>> job.submit()
            >>> result = job.stream_until_complete()
        """
        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = GraphOptimizationJobConfig(
            backend_url=backend_url,
            api_key=api_key,
        )

        # Create delegate using GraphEvolveJob
        from synth_ai.sdk.optimization._impl.graphgen import GraphEvolveJob

        delegate = GraphEvolveJob.from_dataset(
            dataset=dataset,
            policy_models=policy_models,
            rollout_budget=rollout_budget,
            proposer_effort=proposer_effort,
            judge_model=judge_model,
            judge_provider=judge_provider,
            population_size=population_size,
            num_generations=num_generations,
            problem_spec=problem_spec,
            target_llm_calls=target_llm_calls,
            graph_type=graph_type,
            initial_graph_id=initial_graph_id,
            backend_url=backend_url,
            api_key=api_key,
            auto_start=auto_start,
            metadata=metadata,
        )

        return cls(config, _delegate=delegate, _mode="dataset")

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphOptimizationJob:
        """Resume an existing job by ID.

        Args:
            job_id: Existing job ID
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            GraphOptimizationJob instance for the existing job
        """
        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = GraphOptimizationJobConfig(
            backend_url=backend_url,
            api_key=api_key,
        )

        # Use GraphEvolveJob for resume (it handles all job ID types)
        from synth_ai.sdk.optimization._impl.graphgen import GraphEvolveJob

        delegate = GraphEvolveJob.from_job_id(
            job_id=job_id,
            backend_url=backend_url,
            api_key=api_key,
        )

        return cls(config, job_id=job_id, _delegate=delegate, _mode="dataset")

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        if self._job_id:
            return self._job_id
        if self._delegate:
            return self._delegate.job_id
        return None

    @property
    def algorithm(self) -> Algorithm:
        """Get the optimization algorithm."""
        return self._algorithm

    def submit(self) -> str:
        """Submit the job to the backend.

        Returns:
            Job ID

        Raises:
            RuntimeError: If job submission fails
        """
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        if self._delegate is None:
            raise RuntimeError("No delegate configured")

        result = self._delegate.submit()

        # Extract job_id from result
        if isinstance(result, str):
            self._job_id = result
        elif hasattr(result, "graph_evolve_job_id"):
            self._job_id = result.graph_evolve_job_id
        else:
            self._job_id = self._delegate.job_id

        return self._job_id or ""

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Job status dictionary

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return self._delegate.get_status()

    def poll_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 5.0,
        progress: bool = False,
    ) -> GraphOptimizationResult:
        """Poll job until it reaches a terminal state.

        Args:
            timeout: Maximum seconds to wait for job completion
            interval: Seconds between poll attempts
            progress: If True, print status updates during polling

        Returns:
            GraphOptimizationResult with typed status, best_score, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if self._mode == "config":
            # Config mode delegate returns GraphOptimizationResult directly
            result = self._delegate.poll_until_complete(
                timeout=timeout,
                interval=interval,
                progress=progress,
            )
            return GraphOptimizationResult(
                job_id=result.job_id,
                status=JobStatus.from_string(result.status.value),
                algorithm=self._algorithm,
                best_score=result.best_score,
                best_yaml=result.best_yaml,
                best_snapshot_id=result.best_snapshot_id,
                generations_completed=result.generations_completed,
                total_candidates_evaluated=result.total_candidates_evaluated,
                duration_seconds=result.duration_seconds,
                error=result.error,
                raw=result.raw,
            )
        else:
            # Dataset mode - poll via status endpoint
            import time

            start_time = time.time()
            last_data: Dict[str, Any] = {}

            while time.time() - start_time < timeout:
                try:
                    status_data = self.get_status()
                    last_data = status_data
                    status = JobStatus.from_string(status_data.get("status", "pending"))

                    if progress:
                        msg = status_data.get("status", "pending")
                        gen = status_data.get("current_generation") or status_data.get("generation")
                        if gen is not None:
                            msg = f"{msg} | generation {gen}"
                        print(f"[poll] {msg}")

                    if status.is_terminal:
                        return GraphOptimizationResult.from_response(self.job_id or "", last_data)

                except Exception as exc:
                    if progress:
                        print(f"[poll] error: {exc}")

                time.sleep(interval)

            if progress:
                print(f"[poll] timeout after {timeout:.0f}s")

            return GraphOptimizationResult.from_response(self.job_id or "", last_data)

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[StreamHandler]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphOptimizationResult:
        """Stream job events until completion using SSE.

        Args:
            timeout: Maximum seconds to wait
            interval: Seconds between status checks (for SSE reconnects)
            handlers: Optional StreamHandler instances for custom event handling
            on_event: Optional callback called on each event

        Returns:
            GraphOptimizationResult with typed status, best_score, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if self._mode == "config":
            result = self._delegate.stream_until_complete(
                timeout=timeout,
                on_event=on_event,
            )
            return GraphOptimizationResult(
                job_id=result.job_id,
                status=JobStatus.from_string(result.status.value),
                algorithm=self._algorithm,
                best_score=result.best_score,
                best_yaml=result.best_yaml,
                best_snapshot_id=result.best_snapshot_id,
                generations_completed=result.generations_completed,
                total_candidates_evaluated=result.total_candidates_evaluated,
                duration_seconds=result.duration_seconds,
                error=result.error,
                raw=result.raw,
            )
        else:
            final_status = self._delegate.stream_until_complete(
                timeout=timeout,
                interval=interval,
                handlers=handlers,
                on_event=on_event,
            )
            return GraphOptimizationResult.from_response(self.job_id or "", final_status)

    def cancel(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a running job.

        Args:
            reason: Optional reason for cancellation

        Returns:
            Dict with cancellation status

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if self._mode == "config":
            return self._delegate.cancel()
        else:
            return self._delegate.cancel(reason=reason)

    def download_graph_txt(self) -> str:
        """Download a PUBLIC (redacted) graph export for a completed job.

        Returns:
            Graph export text

        Raises:
            RuntimeError: If job hasn't been submitted or delegate doesn't support this
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if hasattr(self._delegate, "download_graph_txt"):
            return self._delegate.download_graph_txt()
        else:
            raise RuntimeError("download_graph_txt is only available for dataset-based jobs")

    def run_inference(
        self,
        input_data: Dict[str, Any],
        *,
        model: Optional[str] = None,
        graph_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference with the optimized graph/workflow.

        Args:
            input_data: Input data matching the task format
            model: Override model (default: use job's policy model)
            graph_snapshot_id: Specific GraphSnapshot to use (default: best)

        Returns:
            Output dictionary

        Raises:
            RuntimeError: If job hasn't been submitted or delegate doesn't support this
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if hasattr(self._delegate, "run_inference"):
            return self._delegate.run_inference(
                input_data,
                model=model,
                graph_snapshot_id=graph_snapshot_id,
            )
        else:
            raise RuntimeError("run_inference is only available for dataset-based jobs")


__all__ = [
    "Algorithm",
    "GraphOptimizationJob",
    "GraphOptimizationJobConfig",
    "GraphOptimizationResult",
    "JobStatus",
]
