"""Research Agent Job SDK.

Provides high-level abstractions for running research agent jobs via the Synth API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

import httpx

# Algorithm types
AlgorithmType = Literal["scaffold_tuning", "evaluation", "trace_analysis"]
BackendType = Literal["daytona", "modal", "docker"]


@dataclass
class ResearchAgentJobConfig:
    """Configuration for a research agent job."""

    algorithm: AlgorithmType

    # Repository (optional if inline_files provided)
    repo_url: str = ""
    repo_branch: str = "main"
    repo_commit: Optional[str] = None

    # Inline files - alternative to repo_url
    # Dict of filepath -> content (e.g., {"pipeline.py": "...", "eval.py": "..."})
    inline_files: Optional[Dict[str, str]] = None

    backend: BackendType = "daytona"
    model: str = "gpt-4o"
    use_synth_proxy: bool = True

    # Algorithm-specific config
    algorithm_config: Dict[str, Any] = field(default_factory=dict)

    # API configuration
    backend_url: str = ""
    api_key: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and resolve defaults."""
        if not self.backend_url:
            self.backend_url = os.environ.get(
                "SYNTH_BACKEND_URL", "https://api.usesynth.ai"
            )
        if not self.api_key:
            self.api_key = os.environ.get("SYNTH_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
            )
        if not self.repo_url and not self.inline_files:
            raise ValueError(
                "Either repo_url or inline_files must be provided"
            )

    @classmethod
    def from_toml(cls, config_path: str | Path) -> ResearchAgentJobConfig:
        """Load configuration from a TOML file.

        Expected TOML structure:
            [research_agent]
            algorithm = "scaffold_tuning"
            repo_url = "https://github.com/your-org/repo"
            repo_branch = "main"
            backend = "daytona"
            model = "gpt-4o"

            [research_agent.scaffold_tuning]
            objective.metric_name = "accuracy"
            objective.max_iterations = 5
            target_files = ["prompts/*.txt"]
            ...
        """
        import tomllib

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        ra_config = data.get("research_agent", {})
        if not ra_config:
            raise ValueError("Config must have [research_agent] section")

        algorithm = ra_config.get("algorithm")
        if not algorithm:
            raise ValueError("research_agent.algorithm is required")

        # Get algorithm-specific config
        algorithm_config = ra_config.get(algorithm, {})

        return cls(
            algorithm=algorithm,
            repo_url=ra_config.get("repo_url", ""),
            repo_branch=ra_config.get("repo_branch", "main"),
            repo_commit=ra_config.get("repo_commit"),
            inline_files=ra_config.get("inline_files"),
            backend=ra_config.get("backend", "daytona"),
            model=ra_config.get("model", "gpt-4o"),
            use_synth_proxy=ra_config.get("use_synth_proxy", True),
            algorithm_config=algorithm_config,
            backend_url=ra_config.get("backend_url", ""),
            api_key=ra_config.get("api_key", ""),
            metadata=ra_config.get("metadata", {}),
        )


@dataclass
class PollOutcome:
    """Result of polling a job."""

    status: str
    data: Dict[str, Any]
    is_terminal: bool = False
    error: Optional[str] = None


class ResearchAgentJobPoller:
    """Poller for research agent jobs."""

    def __init__(self, backend_url: str, api_key: str) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.api_key = api_key

    def poll(self, job_id: str) -> PollOutcome:
        """Poll job status."""
        url = f"{self.backend_url}/api/research-agent/jobs/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            status = data.get("status", "unknown")
            is_terminal = status in ("succeeded", "failed", "canceled")

            return PollOutcome(
                status=status,
                data=data,
                is_terminal=is_terminal,
                error=data.get("error"),
            )
        except httpx.HTTPStatusError as e:
            return PollOutcome(
                status="error",
                data={},
                is_terminal=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return PollOutcome(
                status="error",
                data={},
                is_terminal=False,
                error=str(e),
            )

    def stream_events(
        self, job_id: str, since_seq: int = 0
    ) -> Iterator[Dict[str, Any]]:
        """Stream events from a job."""
        url = f"{self.backend_url}/api/research-agent/jobs/{job_id}/events"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"since_seq": since_seq}

        try:
            response = httpx.get(url, headers=headers, params=params, timeout=30.0)
            response.raise_for_status()
            events = response.json()
            yield from events
        except Exception:
            pass


class ResearchAgentJob:
    """High-level SDK class for running research agent jobs.

    Supports three algorithms:
    - scaffold_tuning: Iteratively improve code/prompts to optimize a metric
    - evaluation: Run evaluation suites across datasets
    - trace_analysis: Analyze past execution traces for patterns

    Example:
        >>> from synth_ai.sdk.api.research_agent import ResearchAgentJob
        >>>
        >>> # From config file
        >>> job = ResearchAgentJob.from_config("my_config.toml")
        >>> job.submit()
        >>> result = job.poll_until_complete()
        >>>
        >>> # Or programmatically
        >>> job = ResearchAgentJob(
        ...     algorithm="scaffold_tuning",
        ...     repo_url="https://github.com/your-org/repo",
        ...     config=ResearchAgentJobConfig(
        ...         algorithm="scaffold_tuning",
        ...         repo_url="https://github.com/your-org/repo",
        ...         algorithm_config={
        ...             "objective": {"metric_name": "accuracy", "max_iterations": 5},
        ...         }
        ...     )
        ... )
        >>> job.submit()
    """

    def __init__(
        self,
        config: ResearchAgentJobConfig,
        job_id: Optional[str] = None,
    ) -> None:
        """Initialize a research agent job.

        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming)
        """
        self.config = config
        self._job_id = job_id
        self._poller = ResearchAgentJobPoller(config.backend_url, config.api_key)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> ResearchAgentJob:
        """Create a research agent job from a TOML config file.
        
        The config file should have a `[research_agent]` section with algorithm,
        repo_url, backend, model, and algorithm-specific configuration.
        
        Args:
            config_path: Path to TOML config file
            backend_url: Override backend URL (defaults to env or production)
            api_key: Override API key (defaults to SYNTH_API_KEY env var)
            
        Returns:
            ResearchAgentJob instance configured from the file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or missing required fields
            
        Example:
            >>> job = ResearchAgentJob.from_config("research_agent_config.toml")
            >>> job_id = job.submit()
        """
        config = ResearchAgentJobConfig.from_toml(config_path)

        if backend_url:
            config.backend_url = backend_url
        if api_key:
            config.api_key = api_key

        return cls(config=config)

    @classmethod
    def from_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> ResearchAgentJob:
        """Resume a job by ID.

        Args:
            job_id: Existing job ID
            backend_url: Backend URL (defaults to env)
            api_key: API key (defaults to env)

        Returns:
            ResearchAgentJob instance
        """
        # Create minimal config for polling (use inline_files placeholder to pass validation)
        config = ResearchAgentJobConfig(
            algorithm="scaffold_tuning",  # Placeholder
            inline_files={"_placeholder": ""},  # Placeholder to pass validation
            backend_url=backend_url or "",
            api_key=api_key or "",
        )
        return cls(config=config, job_id=job_id)

    @classmethod
    def from_files(
        cls,
        files: Dict[str, str],
        algorithm: AlgorithmType = "scaffold_tuning",
        algorithm_config: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o",
        backend: BackendType = "daytona",
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_synth_proxy: bool = True,
    ) -> ResearchAgentJob:
        """Create a job from inline files (no repo required).

        This is a convenience method for running research agent jobs on code
        provided directly as strings, without needing a git repository.

        Args:
            files: Dict of filepath -> content (e.g., {"pipeline.py": "...", "eval.py": "..."})
            algorithm: Algorithm to use (scaffold_tuning, evaluation, trace_analysis)
            algorithm_config: Algorithm-specific configuration
            model: Model for the agent to use
            backend: Container backend (daytona, modal, docker)
            backend_url: Override backend URL
            api_key: Override API key
            use_synth_proxy: Whether to route LLM calls through Synth proxy (default True)

        Returns:
            ResearchAgentJob instance

        Example:
            >>> job = ResearchAgentJob.from_files(
            ...     files={
            ...         "pipeline.py": open("pipeline.py").read(),
            ...         "eval.py": open("eval.py").read(),
            ...     },
            ...     algorithm="scaffold_tuning",
            ...     algorithm_config={
            ...         "objective": {"metric_name": "accuracy", "max_iterations": 5},
            ...         "target_files": ["pipeline.py"],
            ...         "metric": {"metric_type": "custom", "custom_script": "eval.py"},
            ...     },
            ... )
            >>> job.submit()
        """
        config = ResearchAgentJobConfig(
            algorithm=algorithm,
            inline_files=files,
            backend=backend,
            model=model,
            use_synth_proxy=use_synth_proxy,
            algorithm_config=algorithm_config or {},
            backend_url=backend_url or "",
            api_key=api_key or "",
        )
        return cls(config=config)

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID."""
        return self._job_id

    def submit(self) -> str:
        """Submit the job to the backend.

        Returns:
            Job ID

        Raises:
            RuntimeError: If submission fails
        """
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        url = f"{self.config.backend_url.rstrip('/')}/api/research-agent/jobs"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        # Build request payload
        payload: Dict[str, Any] = {
            "algorithm": self.config.algorithm,
            "backend": self.config.backend,
            "model": self.config.model,
            "use_synth_proxy": self.config.use_synth_proxy,
            "metadata": self.config.metadata,
        }

        # Add repo_url if provided
        if self.config.repo_url:
            payload["repo_url"] = self.config.repo_url
            payload["repo_branch"] = self.config.repo_branch
            if self.config.repo_commit:
                payload["repo_commit"] = self.config.repo_commit

        # Add inline_files if provided
        if self.config.inline_files:
            payload["inline_files"] = self.config.inline_files

        # Add algorithm-specific config
        payload[self.config.algorithm] = self.config.algorithm_config

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            self._job_id = data["job_id"]
            return self._job_id
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Failed to submit job: HTTP {e.response.status_code} - {e.response.text[:500]}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to submit job: {e}") from e

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Status dict with keys: status, current_iteration, best_metric_value, etc.

        Raises:
            RuntimeError: If job not submitted
        """
        if not self._job_id:
            raise RuntimeError("Job not submitted yet")

        outcome = self._poller.poll(self._job_id)
        if outcome.error:
            raise RuntimeError(f"Failed to get status: {outcome.error}")
        return outcome.data

    def get_events(self, since_seq: int = 0) -> List[Dict[str, Any]]:
        """Get job events.

        Args:
            since_seq: Return events after this sequence number

        Returns:
            List of event dicts
        """
        if not self._job_id:
            raise RuntimeError("Job not submitted yet")

        return list(self._poller.stream_events(self._job_id, since_seq))

    def poll_until_complete(
        self,
        timeout: float = 3600.0,
        poll_interval: float = 5.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll until job completes.

        Args:
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between polls (seconds)
            on_event: Callback for each new event

        Returns:
            Final job data

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If job fails
        """
        if not self._job_id:
            raise RuntimeError("Job not submitted yet")

        start_time = time.time()
        last_seq = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {self._job_id} timed out after {timeout}s")

            # Get events if callback provided
            if on_event:
                for event in self._poller.stream_events(self._job_id, last_seq):
                    on_event(event)
                    last_seq = max(last_seq, event.get("seq", 0))

            # Check status
            outcome = self._poller.poll(self._job_id)

            if outcome.is_terminal:
                if outcome.status == "failed":
                    raise RuntimeError(
                        f"Job {self._job_id} failed: {outcome.error or 'Unknown error'}"
                    )
                return outcome.data

            time.sleep(poll_interval)

    def cancel(self) -> bool:
        """Cancel the job.

        Returns:
            True if cancellation was requested
        """
        if not self._job_id:
            raise RuntimeError("Job not submitted yet")

        url = f"{self.config.backend_url.rstrip('/')}/api/research-agent/jobs/{self._job_id}/cancel"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            response = httpx.post(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def get_results(self) -> Dict[str, Any]:
        """Get job results (when completed).

        Returns:
            Results dict with metrics, diff, artifacts, etc.
        """
        if not self._job_id:
            raise RuntimeError("Job not submitted yet")

        url = f"{self.config.backend_url.rstrip('/')}/api/research-agent/jobs/{self._job_id}/results"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            response = httpx.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Failed to get results: HTTP {e.response.status_code}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to get results: {e}") from e
