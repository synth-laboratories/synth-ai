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

from .config import (
    DatasetSource,
    GEPAConfig,
    MIPROConfig,
    ModelProvider,
    OptimizationTool,
    PermittedModel,
    PermittedModelsConfig,
    ReasoningEffort,
    ResearchConfig,
)

# Backend type
BackendType = Literal["daytona", "modal", "docker"]


@dataclass
class ResearchAgentJobConfig:
    """Configuration for a research agent job.

    Example:
        >>> config = ResearchAgentJobConfig(
        ...     research=ResearchConfig(
        ...         task_description="Optimize prompt for banking classification",
        ...         tools=[OptimizationTool.MIPRO],
        ...         datasets=[DatasetSource(source_type="huggingface", hf_repo_id="PolyAI/banking77")],
        ...     ),
        ...     repo_url="https://github.com/my-org/my-pipeline",
        ...     model="gpt-5.1-codex-mini",
        ...     max_agent_spend_usd=25.0,
        ... )
    """

    # Research config (typed)
    research: ResearchConfig

    # Repository (optional if inline_files provided)
    repo_url: str = ""
    repo_branch: str = "main"
    repo_commit: Optional[str] = None

    # Inline files - alternative to repo_url
    # Dict of filepath -> content (e.g., {"pipeline.py": "...", "eval.py": "..."})
    inline_files: Optional[Dict[str, str]] = None

    # Execution
    backend: BackendType = "daytona"
    model: str = "gpt-4o"
    use_synth_proxy: bool = True

    # Spend limits
    max_agent_spend_usd: float = 10.0
    """Maximum spend in USD for agent inference and sandbox time. Default: $10."""

    max_synth_spend_usd: float = 100.0
    """Maximum spend in USD for Synth API calls (experiments, evals). Default: $100."""

    # Reasoning effort (for models that support it)
    reasoning_effort: Optional[ReasoningEffort] = None
    """Reasoning effort level: low, medium, high. Only for supported models (o1, o3, gpt-5 family, synth-*)."""

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
            raise ValueError("Either repo_url or inline_files must be provided")

    @classmethod
    def from_toml(cls, config_path: str | Path) -> ResearchAgentJobConfig:
        """Load configuration from a TOML file.

        Expected TOML structure:
            [research_agent]
            repo_url = "https://github.com/your-org/repo"
            repo_branch = "main"
            backend = "daytona"
            model = "gpt-5.1-codex-mini"
            max_agent_spend_usd = 25.0
            max_synth_spend_usd = 150.0
            reasoning_effort = "medium"

            [research_agent.research]
            task_description = "Optimize prompt for accuracy"
            tools = ["mipro"]
            primary_metric = "accuracy"
            num_iterations = 10

            [[research_agent.research.datasets]]
            source_type = "huggingface"
            hf_repo_id = "PolyAI/banking77"

            [research_agent.research.mipro_config]
            meta_model = "llama-3.3-70b-versatile"
            num_trials = 15
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

        # Parse research config
        research_data = ra_config.get("research", {})
        if not research_data:
            raise ValueError("research_agent.research config is required")

        research = _parse_research_config(research_data)

        return cls(
            research=research,
            repo_url=ra_config.get("repo_url", ""),
            repo_branch=ra_config.get("repo_branch", "main"),
            repo_commit=ra_config.get("repo_commit"),
            inline_files=ra_config.get("inline_files"),
            backend=ra_config.get("backend", "daytona"),
            model=ra_config.get("model", "gpt-4o"),
            use_synth_proxy=ra_config.get("use_synth_proxy", True),
            max_agent_spend_usd=ra_config.get("max_agent_spend_usd", 10.0),
            max_synth_spend_usd=ra_config.get("max_synth_spend_usd", 100.0),
            reasoning_effort=ra_config.get("reasoning_effort"),
            backend_url=ra_config.get("backend_url", ""),
            api_key=ra_config.get("api_key", ""),
            metadata=ra_config.get("metadata", {}),
        )


def _parse_research_config(data: Dict[str, Any]) -> ResearchConfig:
    """Parse ResearchConfig from dict (e.g., from TOML)."""
    # Parse tools
    tools_raw = data.get("tools", ["mipro"])
    tools = [
        OptimizationTool(t) if isinstance(t, str) else t
        for t in tools_raw
    ]

    # Parse datasets
    datasets_raw = data.get("datasets", [])
    datasets = [_parse_dataset_source(d) for d in datasets_raw]

    # Parse permitted_models
    permitted_models = None
    if "permitted_models" in data:
        permitted_models = _parse_permitted_models(data["permitted_models"])

    # Parse GEPA config
    gepa_config = None
    if "gepa_config" in data:
        gepa_config = _parse_gepa_config(data["gepa_config"])

    # Parse MIPRO config
    mipro_config = None
    if "mipro_config" in data:
        mipro_config = _parse_mipro_config(data["mipro_config"])

    return ResearchConfig(
        task_description=data.get("task_description", ""),
        tools=tools,
        datasets=datasets,
        primary_metric=data.get("primary_metric", "accuracy"),
        secondary_metrics=data.get("secondary_metrics", []),
        num_iterations=data.get("num_iterations", 10),
        population_size=data.get("population_size", 20),
        timeout_minutes=data.get("timeout_minutes", 60),
        max_eval_samples=data.get("max_eval_samples"),
        permitted_models=permitted_models,
        gepa_config=gepa_config,
        mipro_config=mipro_config,
        initial_prompt=data.get("initial_prompt"),
        pipeline_entrypoint=data.get("pipeline_entrypoint"),
    )


def _parse_dataset_source(data: Dict[str, Any]) -> DatasetSource:
    """Parse DatasetSource from dict."""
    return DatasetSource(
        source_type=data["source_type"],
        description=data.get("description"),
        hf_repo_id=data.get("hf_repo_id"),
        hf_split=data.get("hf_split", "train"),
        hf_subset=data.get("hf_subset"),
        file_ids=data.get("file_ids"),
        inline_data=data.get("inline_data"),
    )


def _parse_permitted_models(data: Dict[str, Any]) -> PermittedModelsConfig:
    """Parse PermittedModelsConfig from dict."""
    models_raw = data.get("models", [])
    models = [
        PermittedModel(
            model=m["model"],
            provider=ModelProvider(m["provider"]) if isinstance(m["provider"], str) else m["provider"],
        )
        for m in models_raw
    ]
    return PermittedModelsConfig(
        models=models,
        default_temperature=data.get("default_temperature", 0.7),
        default_max_tokens=data.get("default_max_tokens", 4096),
    )


def _parse_gepa_config(data: Dict[str, Any]) -> GEPAConfig:
    """Parse GEPAConfig from dict."""
    mutation_provider = data.get("mutation_provider", "groq")
    if isinstance(mutation_provider, str):
        mutation_provider = ModelProvider(mutation_provider)

    return GEPAConfig(
        mutation_model=data.get("mutation_model", "openai/gpt-oss-120b"),
        mutation_provider=mutation_provider,
        mutation_temperature=data.get("mutation_temperature", 0.7),
        mutation_max_tokens=data.get("mutation_max_tokens", 8192),
        population_size=data.get("population_size", 20),
        num_generations=data.get("num_generations", 10),
        elite_fraction=data.get("elite_fraction", 0.2),
        proposer_type=data.get("proposer_type", "dspy"),
        proposer_effort=data.get("proposer_effort", "MEDIUM"),
        proposer_output_tokens=data.get("proposer_output_tokens", "FAST"),
        spec_path=data.get("spec_path"),
        train_size=data.get("train_size"),
        val_size=data.get("val_size"),
        reference_size=data.get("reference_size"),
    )


def _parse_mipro_config(data: Dict[str, Any]) -> MIPROConfig:
    """Parse MIPROConfig from dict."""
    meta_provider = data.get("meta_provider", "groq")
    if isinstance(meta_provider, str):
        meta_provider = ModelProvider(meta_provider)

    return MIPROConfig(
        meta_model=data.get("meta_model", "llama-3.3-70b-versatile"),
        meta_provider=meta_provider,
        meta_temperature=data.get("meta_temperature", 0.7),
        meta_max_tokens=data.get("meta_max_tokens", 4096),
        num_candidates=data.get("num_candidates", 20),
        num_trials=data.get("num_trials", 10),
        proposer_effort=data.get("proposer_effort", "MEDIUM"),
        proposer_output_tokens=data.get("proposer_output_tokens", "FAST"),
        train_size=data.get("train_size"),
        val_size=data.get("val_size"),
        reference_size=data.get("reference_size"),
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

    Research agent jobs use AI to optimize prompts/pipelines using MIPRO or GEPA algorithms.

    Example:
        >>> from synth_ai.sdk.api.research_agent import (
        ...     ResearchAgentJob,
        ...     ResearchAgentJobConfig,
        ...     ResearchConfig,
        ...     DatasetSource,
        ...     OptimizationTool,
        ... )
        >>>
        >>> # Create typed config
        >>> research_config = ResearchConfig(
        ...     task_description="Optimize prompt for banking classification",
        ...     tools=[OptimizationTool.MIPRO],
        ...     datasets=[
        ...         DatasetSource(
        ...             source_type="huggingface",
        ...             hf_repo_id="PolyAI/banking77",
        ...         )
        ...     ],
        ... )
        >>>
        >>> job_config = ResearchAgentJobConfig(
        ...     research=research_config,
        ...     repo_url="https://github.com/my-org/my-pipeline",
        ...     model="gpt-5.1-codex-mini",
        ...     max_agent_spend_usd=25.0,
        ... )
        >>>
        >>> job = ResearchAgentJob(config=job_config)
        >>> job_id = job.submit()
        >>> result = job.poll_until_complete()
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

        Args:
            config_path: Path to TOML config file
            backend_url: Override backend URL (defaults to env or production)
            api_key: Override API key (defaults to SYNTH_API_KEY env var)

        Returns:
            ResearchAgentJob instance configured from the file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or missing required fields
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
        # Create minimal config for polling
        # Use a placeholder ResearchConfig since we're just polling
        research = ResearchConfig(task_description="_placeholder")
        config = ResearchAgentJobConfig(
            research=research,
            inline_files={"_placeholder": ""},
            backend_url=backend_url or "",
            api_key=api_key or "",
        )
        return cls(config=config, job_id=job_id)

    @classmethod
    def from_research_config(
        cls,
        research: ResearchConfig,
        repo_url: str = "",
        repo_branch: str = "main",
        repo_commit: Optional[str] = None,
        inline_files: Optional[Dict[str, str]] = None,
        model: str = "gpt-4o",
        backend: BackendType = "daytona",
        max_agent_spend_usd: float = 10.0,
        max_synth_spend_usd: float = 100.0,
        reasoning_effort: Optional[ReasoningEffort] = None,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_synth_proxy: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchAgentJob:
        """Create a job from a ResearchConfig.

        This is a convenience method for creating jobs programmatically.

        Args:
            research: Research configuration
            repo_url: Git repository URL
            repo_branch: Branch to clone
            repo_commit: Specific commit to checkout
            inline_files: Files to include in workspace
            model: Model for the agent to use
            backend: Container backend (daytona, modal, docker)
            max_agent_spend_usd: Max spend for agent inference
            max_synth_spend_usd: Max spend for Synth API calls
            reasoning_effort: Reasoning effort level (low, medium, high)
            backend_url: Override backend URL
            api_key: Override API key
            use_synth_proxy: Route LLM calls through Synth proxy
            metadata: Additional metadata

        Returns:
            ResearchAgentJob instance
        """
        config = ResearchAgentJobConfig(
            research=research,
            repo_url=repo_url,
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            inline_files=inline_files,
            backend=backend,
            model=model,
            use_synth_proxy=use_synth_proxy,
            max_agent_spend_usd=max_agent_spend_usd,
            max_synth_spend_usd=max_synth_spend_usd,
            reasoning_effort=reasoning_effort,
            backend_url=backend_url or "",
            api_key=api_key or "",
            metadata=metadata or {},
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
            NotImplementedError: If GEPA is requested (not yet supported)
        """
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        # Check for GEPA - not yet fully supported
        if OptimizationTool.GEPA in self.config.research.tools:
            raise NotImplementedError(
                "GEPA optimization is not yet fully supported in the Research Agent SDK. "
                "Please use MIPRO for now. GEPA support is coming soon."
            )

        url = f"{self.config.backend_url.rstrip('/')}/api/research-agent/jobs"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        # Build request payload
        payload: Dict[str, Any] = {
            "algorithm": "research",
            "backend": self.config.backend,
            "model": self.config.model,
            "use_synth_proxy": self.config.use_synth_proxy,
            "max_agent_spend_usd": self.config.max_agent_spend_usd,
            "max_synth_spend_usd": self.config.max_synth_spend_usd,
            "metadata": self.config.metadata,
        }

        # Add reasoning_effort if set
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort

        # Add repo_url if provided
        if self.config.repo_url:
            payload["repo_url"] = self.config.repo_url
            payload["repo_branch"] = self.config.repo_branch
            if self.config.repo_commit:
                payload["repo_commit"] = self.config.repo_commit

        # Add inline_files if provided
        if self.config.inline_files:
            payload["inline_files"] = self.config.inline_files

        # Add research config
        payload["research"] = self.config.research.to_dict()

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
