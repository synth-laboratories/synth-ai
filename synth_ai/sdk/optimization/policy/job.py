"""Policy optimization job implementation.

This module provides the canonical `PolicyOptimizationJob` class for running
policy optimization (prompt/instruction optimization) jobs.

Replaces: `PromptLearningJob` (deprecated)
Backend endpoint: `/api/jobs/{gepa|mipro}` (canonical), `/api/policy-optimization/online/jobs` (legacy)

Algorithms:
- gepa: Genetic Evolutionary Prompt Algorithm (default)
  - Evolutionary algorithm for optimizing prompts through population-based search
  - Uses mutation, crossover, and selection to evolve prompt candidates
  - Supports both online and offline optimization modes

- mipro: Multi-prompt Instruction Proposal Optimizer
  - Systematic instruction proposal and evaluation algorithm
  - Generates new prompt instructions based on reward feedback
  - Supports online mode where you drive rollouts locally
  - Backend provides proxy URL for prompt candidate selection
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

from synth_ai.core.utils.urls import BACKEND_URL_BASE, is_synthtunnel_url
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.optimization.models import PolicyJobStatus as JobStatus
from synth_ai.sdk.optimization.models import PolicyOptimizationResult

if TYPE_CHECKING:
    from synth_ai.core.streaming import StreamHandler


class Algorithm(str, Enum):
    """Supported policy optimization algorithms.

    Attributes:
        GEPA: Genetic Evolutionary Prompt Algorithm - Evolutionary population-based search
        MIPRO: Multi-prompt Instruction Proposal Optimizer - Systematic instruction proposal
    """

    GEPA = "gepa"
    """Genetic Evolutionary Prompt Algorithm - Evolutionary population-based search."""

    MIPRO = "mipro"
    """Multi-prompt Instruction Proposal Optimizer - Systematic instruction proposal."""

    @classmethod
    def from_string(cls, value: str) -> Algorithm:
        """Convert string to Algorithm enum.

        Args:
            value: Algorithm name (case-insensitive)

        Returns:
            Algorithm enum value, defaults to GEPA if invalid
        """
        try:
            return cls(value.lower())
        except ValueError:
            return cls.GEPA  # Default to GEPA


def _extract_task_app_url(payload: Dict[str, Any]) -> Optional[str]:
    section: Any = payload
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


def _load_toml_payload(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]

        return tomllib.loads(path.read_text())
    except Exception:
        try:
            import toml  # type: ignore[import-not-found]

            return toml.loads(path.read_text())
        except Exception:
            return {}


def _infer_task_app_url(config: PolicyOptimizationJobConfig) -> Optional[str]:
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
        payload = _load_toml_payload(config.config_path)
        url = _extract_task_app_url(payload)
        if url:
            return url
    env_url = os.environ.get("TASK_APP_URL", "").strip()
    return env_url or None


@dataclass
class PolicyOptimizationJobConfig:
    """Configuration for a policy optimization job.

    This dataclass holds all the configuration needed to submit and run
    a policy optimization job (GEPA or MIPRO).

    Supports two modes:
    1. **File-based**: Provide `config_path` pointing to a TOML file
    2. **Programmatic**: Provide `config_dict` with the configuration directly

    Attributes:
        config_path: Path to the TOML configuration file. Mutually exclusive with config_dict.
        config_dict: Dictionary with policy optimization configuration.
        backend_url: Base URL of the Synth API backend.
        api_key: Synth API key for authentication.
        localapi_api_key: API key for authenticating with the LocalAPI.
        task_app_worker_token: SynthTunnel worker token for relay auth when using st.usesynth.ai URLs.
        algorithm: Optimization algorithm to use (gepa, mipro).
        allow_experimental: If True, allows use of experimental models.
        overrides: Dictionary of config overrides.

    Example (file-based):
        >>> config = PolicyOptimizationJobConfig(
        ...     config_path=Path("my_config.toml"),
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ... )

    Example (programmatic with GEPA):
        >>> config = PolicyOptimizationJobConfig(
        ...     config_dict={
        ...         "policy_optimization": {
        ...             "algorithm": "gepa",
        ...             "localapi_url": "https://tunnel.example.com",
        ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
        ...             "gepa": {...},
        ...         }
        ...     },
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ... )

    Example (programmatic with MIPRO):
        >>> config = PolicyOptimizationJobConfig(
        ...     config_dict={
        ...         "policy_optimization": {
        ...             "algorithm": "mipro",
        ...             "task_app_url": "https://your-task-app.example.com",
        ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
        ...             "mipro": {
        ...                 "mode": "online",
        ...                 "bootstrap_train_seeds": [0, 1, 2, 3, 4],
        ...                 "val_seeds": [100, 101, 102],
        ...                 "proposer": {"model": "gpt-4o-mini", "provider": "openai"},
        ...             },
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
    localapi_api_key: Optional[str] = None
    task_app_worker_token: Optional[str] = field(default=None, repr=False)
    algorithm: Algorithm = Algorithm.GEPA
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
            self.localapi_api_key = None
        else:
            # Get localapi_api_key from environment if not provided
            if not self.localapi_api_key:
                self.localapi_api_key = ensure_localapi_auth(
                    backend_base=self.backend_url,
                    synth_api_key=self.api_key,
                )

    def to_prompt_learning_config(self) -> Dict[str, Any]:
        """Convert to prompt_learning config format for backward compatibility.

        The backend currently uses 'prompt_learning' section names. This method
        converts our config to that format until the backend is updated.
        """
        if self.config_dict:
            # Check for policy_optimization section and convert to prompt_learning
            if "policy_optimization" in self.config_dict:
                config = dict(self.config_dict)
                config["prompt_learning"] = config.pop("policy_optimization")
                # Also convert localapi_url to task_app_url for backend compat
                if "localapi_url" in config["prompt_learning"]:
                    config["prompt_learning"]["task_app_url"] = config["prompt_learning"].pop(
                        "localapi_url"
                    )
                return config
            return self.config_dict
        return {}


class PolicyOptimizationJob:
    """High-level SDK class for running policy optimization jobs.

    This is the canonical class for policy optimization, replacing
    `PromptLearningJob`. It supports both GEPA and MIPRO algorithms.

    **GEPA** (Genetic Evolutionary Prompt Algorithm):
        - Evolutionary algorithm using population-based search
        - Optimizes prompts through mutation, crossover, and selection
        - Supports both online and offline optimization modes
        - Best for: Comprehensive search across prompt space

    **MIPRO** (Multi-prompt Instruction Proposal Optimizer):
        - Systematic instruction proposal and evaluation
        - Generates new prompt instructions based on reward feedback
        - Online mode: You drive rollouts, backend provides prompt candidates
        - Best for: Iterative refinement with real-time prompt evolution

    Example (GEPA):
        >>> from synth_ai.sdk.optimization.policy import PolicyOptimizationJob
        >>>
        >>> # Create job from config
        >>> job = PolicyOptimizationJob.from_config(
        ...     config_path="gepa_config.toml",
        ...     api_key=os.environ["SYNTH_API_KEY"],
        ...     algorithm="gepa"
        ... )
        >>>
        >>> # Submit job
        >>> job_id = job.submit()
        >>> print(f"Job submitted: {job_id}")
        >>>
        >>> # Stream until complete (recommended)
        >>> result = job.stream_until_complete()
        >>> print(f"Best score: {result.best_score}")

    Example (MIPRO):
        >>> from synth_ai.sdk.optimization.policy import PolicyOptimizationJob
        >>>
        >>> # Create MIPRO job from config
        >>> job = PolicyOptimizationJob.from_config(
        ...     config_path="mipro_config.toml",
        ...     api_key=os.environ["SYNTH_API_KEY"],
        ...     algorithm="mipro"
        ... )
        >>>
        >>> # Submit job
        >>> job_id = job.submit()
        >>>
        >>> # Poll until complete
        >>> result = job.poll_until_complete(timeout=3600.0)
        >>> print(f"Best score: {result.best_score}")

    Attributes:
        job_id: The job ID (None until submitted)
        algorithm: The optimization algorithm being used (GEPA or MIPRO)
    """

    def __init__(
        self,
        config: PolicyOptimizationJobConfig,
        job_id: Optional[str] = None,
        skip_health_check: bool = False,
    ) -> None:
        """Initialize a policy optimization job.

        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
            skip_health_check: If True, skip LocalAPI health check before submission.
        """
        self.config = config
        self._job_id = job_id
        self._skip_health_check = skip_health_check
        self._algorithm = config.algorithm

        # Internally delegate to PromptLearningJob for backend compatibility
        # This will be removed once the backend supports /api/policy-optimization endpoints
        self._delegate: Optional[Any] = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        localapi_api_key: Optional[str] = None,
        task_app_worker_token: Optional[str] = None,
        algorithm: str | Algorithm = Algorithm.GEPA,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PolicyOptimizationJob:
        """Create a job from a TOML config file.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            localapi_api_key: LocalAPI key (defaults to ENVIRONMENT_API_KEY env var)
            task_app_worker_token: SynthTunnel worker token for relay auth
            algorithm: Optimization algorithm (gepa or mipro)
            allow_experimental: Allow experimental models
            overrides: Config overrides

        Returns:
            PolicyOptimizationJob instance

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

        if isinstance(algorithm, str):
            algorithm = Algorithm.from_string(algorithm)

        config = PolicyOptimizationJobConfig(
            config_path=config_path_obj,
            backend_url=backend_url,
            api_key=api_key,
            localapi_api_key=localapi_api_key,
            task_app_worker_token=task_app_worker_token,
            algorithm=algorithm,
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
        localapi_api_key: Optional[str] = None,
        task_app_worker_token: Optional[str] = None,
        algorithm: str | Algorithm = Algorithm.GEPA,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
        skip_health_check: bool = False,
    ) -> PolicyOptimizationJob:
        """Create a job from a configuration dictionary.

        The config_dict can use either the new 'policy_optimization' section
        or the legacy 'prompt_learning' section for backward compatibility.

        Args:
            config_dict: Configuration dictionary
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            localapi_api_key: LocalAPI key (defaults to ENVIRONMENT_API_KEY env var)
            task_app_worker_token: SynthTunnel worker token for relay auth
            algorithm: Optimization algorithm (gepa or mipro)
            allow_experimental: Allow experimental models
            overrides: Config overrides
            skip_health_check: If True, skip LocalAPI health check

        Returns:
            PolicyOptimizationJob instance

        Example (GEPA):
            >>> job = PolicyOptimizationJob.from_dict(
            ...     config_dict={
            ...         "policy_optimization": {
            ...             "algorithm": "gepa",
            ...             "localapi_url": "https://tunnel.example.com",
            ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
            ...             "gepa": {...},
            ...         }
            ...     },
            ...     api_key="sk_live_...",
            ... )

        Example (MIPRO):
            >>> job = PolicyOptimizationJob.from_dict(
            ...     config_dict={
            ...         "policy_optimization": {
            ...             "algorithm": "mipro",
            ...             "task_app_url": "https://your-task-app.example.com",
            ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
            ...             "mipro": {
            ...                 "mode": "online",
            ...                 "bootstrap_train_seeds": [0, 1, 2, 3, 4],
            ...                 "val_seeds": [100, 101, 102],
            ...                 "proposer": {"model": "gpt-4o-mini", "provider": "openai"},
            ...             },
            ...         }
            ...     },
            ...     api_key="sk_live_...",
            ... )
        """
        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        if isinstance(algorithm, str):
            algorithm = Algorithm.from_string(algorithm)

        config = PolicyOptimizationJobConfig(
            config_dict=config_dict,
            backend_url=backend_url,
            api_key=api_key,
            localapi_api_key=localapi_api_key,
            task_app_worker_token=task_app_worker_token,
            algorithm=algorithm,
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        )

        # Auto-detect tunnel URLs and skip health check
        if skip_health_check is False:
            section = config_dict.get("policy_optimization") or config_dict.get(
                "prompt_learning", {}
            )
            localapi_url = (
                section.get("localapi_url")
                or section.get("task_app_url")
                or section.get("local_api_url")
            )
            if localapi_url and (
                ".trycloudflare.com" in localapi_url.lower()
                or ".cfargotunnel.com" in localapi_url.lower()
            ):
                skip_health_check = True

        return cls(config, skip_health_check=skip_health_check)

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> PolicyOptimizationJob:
        """Resume an existing job by ID.

        Args:
            job_id: Existing job ID
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            PolicyOptimizationJob instance for the existing job
        """
        if not backend_url:
            backend_url = BACKEND_URL_BASE

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        # Create minimal config for resume mode
        config = PolicyOptimizationJobConfig(
            config_dict={"policy_optimization": {"_resumed": True}},
            backend_url=backend_url,
            api_key=api_key,
        )

        return cls(config, job_id=job_id)

    def _get_delegate(self) -> Any:
        """Get or create the internal PromptLearningJob delegate.

        This provides backward compatibility with the current backend endpoints.
        Will be removed when backend supports /api/policy-optimization endpoints.
        """
        if self._delegate is None:
            from synth_ai.sdk.optimization.clients.jobs import (
                PromptLearningJob,
                PromptLearningJobConfig,
            )

            # Convert config to prompt_learning format
            if self.config.config_path:
                delegate_config = PromptLearningJobConfig(
                    config_path=self.config.config_path,
                    backend_url=self.config.backend_url,
                    api_key=self.config.api_key,
                    task_app_api_key=self.config.localapi_api_key,
                    task_app_worker_token=self.config.task_app_worker_token,
                    allow_experimental=self.config.allow_experimental,
                    overrides=self.config.overrides,
                )
            else:
                # Convert policy_optimization -> prompt_learning in config dict
                config_dict = self.config.to_prompt_learning_config()
                delegate_config = PromptLearningJobConfig(
                    config_dict=config_dict,
                    backend_url=self.config.backend_url,
                    api_key=self.config.api_key,
                    task_app_api_key=self.config.localapi_api_key,
                    task_app_worker_token=self.config.task_app_worker_token,
                    allow_experimental=self.config.allow_experimental,
                    overrides=self.config.overrides,
                )

            self._delegate = PromptLearningJob(
                delegate_config,
                job_id=self._job_id,
                skip_health_check=self._skip_health_check,
            )

        return self._delegate

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        return self._job_id

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
            ValueError: If LocalAPI health check fails
        """
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        # Delegate to PromptLearningJob for now
        delegate = self._get_delegate()
        self._job_id = delegate.submit()
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

        delegate = self._get_delegate()
        return delegate.get_status()

    def poll_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        progress: bool = False,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_timeout: float = 180.0,
    ) -> PolicyOptimizationResult:
        """Poll job until it reaches a terminal state.

        Args:
            timeout: Maximum seconds to wait for job completion
            interval: Seconds between poll attempts
            progress: If True, print status updates during polling
            on_status: Optional callback called on each status update
            request_timeout: HTTP timeout for each status request

        Returns:
            PolicyOptimizationResult with typed status, best_score, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout is exceeded
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        delegate = self._get_delegate()
        pl_result = delegate.poll_until_complete(
            timeout=timeout,
            interval=interval,
            progress=progress,
            on_status=on_status,
            request_timeout=request_timeout,
        )

        # Convert PromptLearningResult to PolicyOptimizationResult
        return PolicyOptimizationResult(
            job_id=pl_result.job_id,
            status=pl_result.status,
            algorithm=str(self._algorithm),
            best_reward=pl_result.best_reward,
            best_prompt=pl_result.best_prompt,
            error=pl_result.error,
            raw=pl_result.raw,
        )

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[StreamHandler]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PolicyOptimizationResult:
        """Stream job events until completion using SSE.

        This provides real-time event streaming instead of polling,
        reducing server load and providing faster updates.

        Args:
            timeout: Maximum seconds to wait (default: 3600 = 1 hour)
            interval: Seconds between status checks (for SSE reconnects)
            handlers: Optional StreamHandler instances for custom event handling
            on_event: Optional callback called on each event

        Returns:
            PolicyOptimizationResult with typed status, best_score, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        delegate = self._get_delegate()
        pl_result = delegate.stream_until_complete(
            timeout=timeout,
            interval=interval,
            handlers=handlers,
            on_event=on_event,
        )

        # Convert PromptLearningResult to PolicyOptimizationResult
        return PolicyOptimizationResult(
            job_id=pl_result.job_id,
            status=pl_result.status,
            algorithm=str(self._algorithm),
            best_reward=pl_result.best_reward,
            best_prompt=pl_result.best_prompt,
            error=pl_result.error,
            raw=pl_result.raw,
        )

    def get_results(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.).

        Returns:
            Results dictionary with best_prompt, best_score, etc.

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        delegate = self._get_delegate()
        return delegate.get_results()

    def get_best_prompt_text(self, rank: int = 1) -> Optional[str]:
        """Get the text of the best prompt by rank.

        Args:
            rank: Prompt rank (1 = best, 2 = second best, etc.)

        Returns:
            Prompt text or None if not found
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        delegate = self._get_delegate()
        return delegate.get_best_prompt_text(rank=rank)

    def cancel(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a running job.

        Args:
            reason: Optional reason for cancellation

        Returns:
            Dict with cancellation status

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        delegate = self._get_delegate()
        return delegate.cancel(reason=reason)

    def query_workflow_state(self) -> Dict[str, Any]:
        """Query the Temporal workflow state for instant polling.

        Returns:
            Dict with workflow state

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        delegate = self._get_delegate()
        return delegate.query_workflow_state()


__all__ = [
    "Algorithm",
    "JobStatus",
    "PolicyOptimizationJob",
    "PolicyOptimizationJobConfig",
    "PolicyOptimizationResult",
]
