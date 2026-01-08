"""First-class SDK API for GraphGen (Automated Design of Agentic Systems).

GraphGen is a simplified "Workflows API" for prompt optimization that:
- Uses a simple JSON dataset format (GraphGenTaskSet) instead of TOML configs
- Auto-generates task apps from the dataset (no user-managed task apps)
- Has built-in judge configurations (rubric, contrastive, gold_examples)
- Wraps GEPA internally for the actual optimization

Example CLI usage:
    uvx synth-ai train --type adas --dataset my_tasks.json --poll

Example SDK usage:
    from synth_ai.sdk.api.train.graphgen import GraphGenJob
    from synth_ai.sdk.api.train.graphgen_models import GraphGenTaskSet, GraphGenTask

    # From a dataset file
    job = GraphGenJob.from_dataset("my_tasks.json")
    job.submit()
    result = job.stream_until_complete()
    print(f"Best score: {result.get('best_score')}")

    # Or programmatically
    dataset = GraphGenTaskSet(
        metadata=GraphGenTaskSetMetadata(name="My Tasks"),
        initial_prompt="You are a helpful assistant...",
        tasks=[GraphGenTask(id="t1", input={"question": "What is 2+2?"})],
        gold_outputs=[GraphGenGoldOutput(output={"answer": "4"}, task_id="t1")],
    )
    job = GraphGenJob.from_dataset(dataset, policy_models="gpt-4o-mini")
    job.submit()
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from synth_ai.core.telemetry import log_info

from .graphgen_models import (
    GraphGenJobConfig,
    GraphGenTaskSet,
    load_graphgen_taskset,
    parse_graphgen_taskset,
    # Aliases for new names
    GraphGenJobConfig,
    GraphGenTaskSet,
    load_graphgen_taskset,
    parse_graphgen_taskset,
)
from .utils import ensure_api_base, http_get, http_post


@dataclass
class GraphGenJobResult:
    """Result from an GraphGen job."""

    graphgen_job_id: str
    status: str
    best_score: Optional[float] = None
    best_snapshot_id: Optional[str] = None
    error: Optional[str] = None
    dataset_name: Optional[str] = None
    task_count: Optional[int] = None
    graph_evolve_job_id: Optional[str] = None


@dataclass
class GraphGenSubmitResult:
    """Result from submitting an GraphGen job."""

    graphgen_job_id: str
    status: str
    dataset_name: str
    task_count: int
    rollout_budget: int
    policy_models: List[str]
    judge_mode: str
    graph_evolve_job_id: Optional[str] = None


class GraphGenJob:
    """High-level SDK class for running GraphGen workflow optimization jobs.

    GraphGen (Automated Design of Agentic Systems) provides a simplified API for
    graph/workflow optimization that doesn't require users to manage task apps.

    Key differences from PromptLearningJob:
    - Uses JSON dataset format (GraphGenTaskSet) instead of TOML configs
    - No task app management required - GraphGen builds it internally
    - Built-in judge modes (rubric, contrastive, gold_examples)
    - Graph-first: trains multi-node workflows by default (Graph-GEPA)
    - Public graph downloads are redacted `.txt` exports only
    - Simpler configuration with sensible defaults

    Example:
        >>> from synth_ai.sdk.api.train.adas import GraphGenJob
        >>>
        >>> # Create job from dataset file
        >>> job = GraphGenJob.from_dataset(
        ...     dataset="my_tasks.json",
        ...     policy_models="gpt-4o-mini",
        ...     rollout_budget=100,
        ... )
        >>>
        >>> # Submit and stream
        >>> job.submit()
        >>> result = job.stream_until_complete(timeout=3600.0)
        >>> print(f"Best score: {result.get('best_score')}")
        >>> 
        >>> # Download public graph export
        >>> export_txt = job.download_graph_txt()
        >>> print(export_txt)
        >>>
        >>> # Run inference with optimized prompt
        >>> output = job.run_inference({"question": "What is 2+2?"})
    """

    def __init__(
        self,
        *,
        dataset: GraphGenTaskSet,
        config: GraphGenJobConfig,
        backend_url: str,
        api_key: str,
        auto_start: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize an GraphGen job.

        Args:
            dataset: The GraphGenTaskSet containing tasks and evaluation config
            config: Job configuration (policy model, budget, etc.)
            backend_url: Backend API URL
            api_key: Synth API key
            auto_start: Whether to start the job immediately after creation
            metadata: Additional metadata for the job
        """
        self.dataset = dataset
        self.config = config
        self.backend_url = ensure_api_base(backend_url)
        self.api_key = api_key
        self.auto_start = auto_start
        self.metadata = metadata or {}

        self._graphgen_job_id: Optional[str] = None
        self._graph_evolve_job_id: Optional[str] = None
        self._submit_result: Optional[GraphGenSubmitResult] = None

    @classmethod
    def from_dataset(
        cls,
        dataset: str | Path | Dict[str, Any] | GraphGenTaskSet,
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
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_start: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphGenJob:
        """Create an GraphGen job from a dataset.

        Args:
            dataset: Dataset as file path, dict, or GraphGenTaskSet object
            policy_models: Model(s) to use for policy inference. Can be a single string
                (will be converted to a one-element list) or a list of strings.
            rollout_budget: Total number of rollouts for optimization
            proposer_effort: Proposer effort level (low, medium, high)
            judge_model: Override judge model from dataset
            judge_provider: Override judge provider from dataset
            population_size: Population size for GEPA
            num_generations: Number of generations (auto-calculated if not specified)
            problem_spec: Detailed problem specification for the graph proposer.
                Include domain-specific info like valid output labels for classification.
            target_llm_calls: Target number of LLM calls for the graph (1-10).
                Controls how many LLM nodes the graph should use. Defaults to 5.
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            auto_start: Whether to start the job immediately
            metadata: Additional metadata for the job

        Returns:
            GraphGenJob instance

        Example:
            >>> # From file with single model
            >>> job = GraphGenJob.from_dataset("tasks.json", policy_models="gpt-4o-mini")
            >>>
            >>> # From file with multiple models
            >>> job = GraphGenJob.from_dataset("tasks.json", policy_models=["gpt-4o-mini", "gpt-4.1-mini"])
            >>>
            >>> # From dict
            >>> job = GraphGenJob.from_dataset({
            ...     "metadata": {"name": "My Tasks"},
            ...     "initial_prompt": "You are helpful.",
            ...     "tasks": [{"id": "t1", "input": {"q": "Hi"}}],
            ... })
            >>>
            >>> # From GraphGenTaskSet object
            >>> job = GraphGenJob.from_dataset(my_taskset, policy_models=["gpt-4o"])
        """
        from synth_ai.core.env import get_backend_from_env

        # Parse dataset
        if isinstance(dataset, (str, Path)):
            parsed_dataset = load_graphgen_taskset(dataset)
        elif isinstance(dataset, dict):
            parsed_dataset = parse_graphgen_taskset(dataset)
        elif isinstance(dataset, GraphGenTaskSet):
            parsed_dataset = dataset
        else:
            raise TypeError(
                f"dataset must be a file path, dict, or GraphGenTaskSet, got {type(dataset)}"
            )

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

        # Normalize policy_models: convert single string to list
        if isinstance(policy_models, str):
            policy_models_list = [policy_models]
        else:
            policy_models_list = list(policy_models)
        
        if not policy_models_list:
            raise ValueError("policy_models must contain at least one model")
        
        # Build config
        config = GraphGenJobConfig(
            policy_models=policy_models_list,
            rollout_budget=rollout_budget,
            proposer_effort=proposer_effort,
            verifier_model=judge_model,
            verifier_provider=judge_provider,
            population_size=population_size,
            num_generations=num_generations,
            problem_spec=problem_spec,
            target_llm_calls=target_llm_calls,
        )

        return cls(
            dataset=parsed_dataset,
            config=config,
            backend_url=backend_url,
            api_key=api_key,
            auto_start=auto_start,
            metadata=metadata,
        )

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphGenJob:
        """Resume an existing GraphGen job by ID.

        Args:
            job_id: GraphGen job ID ("graphgen_*") or underlying GEPA job ID ("pl_*")
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            GraphGenJob instance for the existing job
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

        # Create minimal instance - dataset will be fetched from backend if needed
        # For now, create a placeholder dataset
        from .graphgen_models import GraphGenTaskSetMetadata, GraphGenTask
        placeholder_dataset = GraphGenTaskSet(
            metadata=GraphGenTaskSetMetadata(name="(resumed job)"),
            tasks=[GraphGenTask(id="placeholder", input={})],
        )

        job = cls(
            dataset=placeholder_dataset,
            config=GraphGenJobConfig(policy_models=["(resumed)"]),  # Placeholder, will be fetched from backend
            backend_url=backend_url,
            api_key=api_key,
            auto_start=False,
        )

        # Accept GraphGen/GraphGen or graph_evolve/GEPA job IDs - backend handles resolution internally
        valid_prefixes = ("graphgen_", "graphgen_", "graph_evolve_", "graph_evolve_", "pl_")
        if not any(job_id.startswith(p) for p in valid_prefixes):
            raise ValueError(
                f"Unsupported job ID format: {job_id!r}. "
                f"Expected one of: {valid_prefixes}"
            )
        job._graphgen_job_id = job_id
        if job_id.startswith("pl_"):
            job._graph_evolve_job_id = job_id
        return job

    @classmethod
    def from_graph_evolve_job_id(
        cls,
        graph_evolve_job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphGenJob:
        """Alias for resuming an GraphGen job from a GEPA job ID."""
        return cls.from_job_id(graph_evolve_job_id, backend_url=backend_url, api_key=api_key)

    @property
    def job_id(self) -> Optional[str]:
        """Get the GraphGen job ID (None if not yet submitted)."""
        return self._graphgen_job_id

    @property
    def graph_evolve_job_id(self) -> Optional[str]:
        """Get the underlying GEPA job ID if known."""
        if self._graph_evolve_job_id:
            return self._graph_evolve_job_id
        if self._submit_result and self._submit_result.graph_evolve_job_id:
            return self._submit_result.graph_evolve_job_id
        return None

    def _build_payload(self) -> Dict[str, Any]:
        """Build the job creation payload."""
        # Merge config num_generations into metadata if provided
        metadata = dict(self.metadata) if self.metadata else {}
        if self.config.num_generations is not None:
            metadata["num_generations"] = self.config.num_generations
        if self.config.population_size != 4:  # Only include if non-default
            metadata["population_size"] = self.config.population_size
        if self.config.num_parents != 2:
            metadata["num_parents"] = self.config.num_parents
        if self.config.evaluation_seeds is not None:
            metadata["evaluation_seeds"] = self.config.evaluation_seeds

        # Extract eval/feedback sample sizes from metadata as direct fields
        eval_sample_size = metadata.pop("eval_sample_size", None)
        feedback_sample_size = metadata.pop("feedback_sample_size", None)

        # Build dataset dict and ensure it has an initial_prompt to satisfy legacy backend validation
        # GraphGen is graph-first and doesn't really use this, so we use a placeholder or problem_spec
        dataset_dict = self.dataset.model_dump(mode="json", exclude_none=False)
        if "initial_prompt" not in dataset_dict:
            dataset_dict["initial_prompt"] = self.config.problem_spec or "Optimizing prompt graph..."
        
        # Ensure verifier_config is properly included
        if "verifier_config" not in dataset_dict or dataset_dict.get("verifier_config") is None:
            # Fallback: create verifier_config from dataset's verifier_config
            if self.dataset.verifier_config:
                dataset_dict["verifier_config"] = self.dataset.verifier_config.model_dump(mode="json")

        payload: Dict[str, Any] = {
            "dataset": dataset_dict,
            "initial_prompt": None,  # Top-level initial_prompt is ignored in favor of dataset.initial_prompt
            "policy_models": self.config.policy_models,
            "policy_provider": self.config.policy_provider,
            "rollout_budget": self.config.rollout_budget,
            "proposer_effort": self.config.proposer_effort,
            "judge_model": getattr(self.config, "verifier_model", None),
            "judge_provider": getattr(self.config, "verifier_provider", None),
            "problem_spec": self.config.problem_spec,
            "target_llm_calls": self.config.target_llm_calls,
            "eval_sample_size": eval_sample_size,
            "feedback_sample_size": feedback_sample_size,
            "metadata": metadata,
            "auto_start": self.auto_start,
        }

        # Strip unset optional fields so we don't send nulls to strict backends.
        if payload.get("eval_sample_size") is None:
            payload.pop("eval_sample_size", None)
        if payload.get("feedback_sample_size") is None:
            payload.pop("feedback_sample_size", None)
        if payload.get("policy_provider") is None:
            payload.pop("policy_provider", None)
        if payload.get("judge_model") is None:
            payload.pop("judge_model", None)
        if payload.get("judge_provider") is None:
            payload.pop("judge_provider", None)
        if payload.get("problem_spec") is None:
            payload.pop("problem_spec", None)
        if payload.get("target_llm_calls") is None:
            payload.pop("target_llm_calls", None)

        return payload

    def submit(self) -> GraphGenSubmitResult:
        """Submit the job to the backend.

        Returns:
            GraphGenSubmitResult with job IDs and initial status

        Raises:
            RuntimeError: If job submission fails
        """
        from .graphgen_validators import validate_graphgen_job_config

        ctx: Dict[str, Any] = {"dataset_name": self.dataset.metadata.name}
        log_info("GraphGenJob.submit invoked", ctx=ctx)

        if self._graphgen_job_id:
            raise RuntimeError(f"Job already submitted: {self._graphgen_job_id}")

        # Validate config + dataset before expensive API call.
        validate_graphgen_job_config(self.config, self.dataset)

        payload = self._build_payload()

        # Submit job - use /graphgen/jobs endpoint (legacy: /adas/jobs)
        create_url = f"{self.backend_url}/graphgen/jobs"
        
        # Debug: print payload for troubleshooting
        import json
        debug_payload = {k: v for k, v in payload.items() if k != "dataset"}
        debug_payload["dataset"] = {"metadata": payload["dataset"].get("metadata", {}) if isinstance(payload["dataset"], dict) else "..."}
        log_info(f"Submitting GraphGen job payload (excluding dataset): {json.dumps(debug_payload, indent=2)}")
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Submitting GraphGen job to: {create_url}")

        resp = http_post(create_url, headers=headers, json_body=payload, timeout=180.0)

        if resp.status_code not in (200, 201):
            error_msg = f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
            if resp.status_code == 404:
                error_msg += (
                    f"\n\nPossible causes:"
                    f"\n1. Backend route /api/graphgen/jobs not registered"
                    f"\n2. GraphGen feature may not be enabled on this backend"
                    f"\n3. Verify backend is running at: {self.backend_url}"
                )
            raise RuntimeError(error_msg)

        try:
            js = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}") from e

        self._graphgen_job_id = js.get("graphgen_job_id")

        if not self._graphgen_job_id:
            raise RuntimeError("Response missing graphgen_job_id")

        self._graph_evolve_job_id = js.get("graph_evolve_job_id")

        # Get judge_mode from response or fallback to dataset verifier_config
        judge_mode = js.get("judge_mode")
        if not judge_mode and self.dataset.verifier_config:
            judge_mode = self.dataset.verifier_config.mode
        if not judge_mode:
            judge_mode = "rubric"  # Default fallback
        
        # Extract policy_models from response (backend may return policy_model for backward compat)
        policy_models_response = js.get("policy_models")
        if not policy_models_response:
            # Backward compatibility: if backend returns policy_model, convert to list
            policy_model_single = js.get("policy_model")
            if policy_model_single:
                policy_models_response = [policy_model_single]
            else:
                policy_models_response = self.config.policy_models
        
        self._submit_result = GraphGenSubmitResult(
            graphgen_job_id=self._graphgen_job_id,
            status=js.get("status", "queued"),
            dataset_name=js.get("dataset_name", self.dataset.metadata.name),
            task_count=js.get("task_count", len(self.dataset.tasks)),
            rollout_budget=js.get("rollout_budget", self.config.rollout_budget),
            policy_models=policy_models_response,
            judge_mode=judge_mode,
            graph_evolve_job_id=self._graph_evolve_job_id,
        )

        ctx["graphgen_job_id"] = self._graphgen_job_id
        log_info("GraphGenJob.submit completed", ctx=ctx)

        return self._submit_result

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Job status dictionary

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self.backend_url}/graphgen/jobs/{self.job_id}"
        headers = {
            "X-API-Key": self.api_key,
        }

        resp = http_get(url, headers=headers, timeout=30.0)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to get job status: {resp.status_code} - {resp.text[:500]}"
            )

        data = resp.json()
        gepa_id = data.get("graph_evolve_job_id")
        if gepa_id:
            self._graph_evolve_job_id = gepa_id
        return data

    def start(self) -> Dict[str, Any]:
        """Start a queued GraphGen job.

        This is only needed if the job was created with auto_start=False or ended up queued.
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self.backend_url}/graphgen/jobs/{self.job_id}/start"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        resp = http_post(url, headers=headers, json_body=None, timeout=60.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to start job: {resp.status_code} - {resp.text[:500]}"
            )
        data = resp.json()
        if self._submit_result and "status" in data:
            self._submit_result.status = data.get("status", self._submit_result.status)
        return data

    def get_events(self, *, since_seq: int = 0, limit: int = 1000) -> Dict[str, Any]:
        """Fetch events for this GraphGen job.

        Returns backend envelope: {"events": [...], "has_more": bool, "next_seq": int}.
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        base = f"{self.backend_url}/graphgen/jobs/{self.job_id}/events"
        url = f"{base}?since_seq={since_seq}&limit={limit}"
        headers = {"X-API-Key": self.api_key}

        resp = http_get(url, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to get events: {resp.status_code} - {resp.text[:500]}"
            )
        return resp.json()

    def get_metrics(
        self,
        *,
        name: Optional[str] = None,
        after_step: Optional[int] = None,
        limit: int = 500,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch metrics for this GraphGen job.

        Mirrors GET /api/graphgen/jobs/{job_id}/metrics.
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        from urllib.parse import urlencode

        params: Dict[str, Any] = {"limit": limit}
        if name is not None:
            params["name"] = name
        if after_step is not None:
            params["after_step"] = after_step
        if run_id is not None:
            params["run_id"] = run_id

        qs = urlencode(params)
        url = f"{self.backend_url}/graphgen/jobs/{self.job_id}/metrics?{qs}"
        headers = {"X-API-Key": self.api_key}

        resp = http_get(url, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to get metrics: {resp.status_code} - {resp.text[:500]}"
            )
        return resp.json()

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 5.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Stream job events until completion using SSE.

        Args:
            timeout: Maximum seconds to wait
            interval: Seconds between status checks (for SSE reconnects)
            handlers: Optional StreamHandler instances for custom event handling
            on_event: Optional callback called on each event

        Returns:
            Final job status dictionary

        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout exceeded
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        from synth_ai.sdk.streaming import (
            GraphGenHandler,
            JobStreamer,
            StreamConfig,
            StreamEndpoints,
            StreamType,
        )

        # Build stream config
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            max_events_per_poll=500,
            deduplicate=True,
        )

        # Use provided handlers or default CLI handler
        if handlers is None:
            handlers = [GraphGenHandler()]

        # Create streamer with GraphGen endpoints
        # Backend handles GraphGen → GEPA resolution internally via job_relationships table
        streamer = JobStreamer(
            base_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,  # Only GraphGen job ID - backend resolves to GEPA internally
            endpoints=StreamEndpoints.graphgen(self.job_id),
            config=config,
            handlers=list(handlers),
            interval_seconds=interval,
            timeout_seconds=timeout,
        )

        # Run streaming
        final_status = asyncio.run(streamer.stream_until_terminal())

        return final_status

    def download_prompt(self) -> str:
        """Download the optimized prompt from a completed job.

        For graph-first jobs, prefer `download_graph_txt()`; this method is
        mainly useful for legacy single-node prompt workflows.

        Returns:
            Optimized prompt text

        Raises:
            RuntimeError: If job hasn't been submitted or isn't complete
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self.backend_url}/graphgen/jobs/{self.job_id}/download"
        headers = {
            "X-API-Key": self.api_key,
        }

        resp = http_get(url, headers=headers, timeout=30.0)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download prompt: {resp.status_code} - {resp.text[:500]}"
            )

        data = resp.json()
        return data.get("prompt", "")

    def download_graph_txt(self) -> str:
        """Download a PUBLIC (redacted) graph export for a completed job.

        Graph-first GraphGen jobs produce multi-node graphs. The internal graph
        YAML/spec is proprietary and never exposed. This helper downloads the
        `.txt` export from:
            GET /api/graphgen/jobs/{job_id}/graph.txt
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self.backend_url}/graphgen/jobs/{self.job_id}/graph.txt"
        headers = {"X-API-Key": self.api_key}

        resp = http_get(url, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download graph export: {resp.status_code} - {resp.text[:500]}"
            )
        return resp.text

    def run_inference(
        self,
        input_data: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt_snapshot_id: Optional[str] = None,
        graph_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference with the optimized graph/workflow.

        Args:
            input_data: Input data matching the task format
            model: Override model (default: use job's policy model)
            prompt_snapshot_id: Legacy alias for selecting a specific snapshot.
            graph_snapshot_id: Specific GraphSnapshot to use (default: best).
                Preferred for graph-first jobs. If provided, it is sent as
                `prompt_snapshot_id` for backward-compatible backend routing.

        Returns:
            Output dictionary

        Raises:
            RuntimeError: If job hasn't been submitted
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if prompt_snapshot_id and graph_snapshot_id:
            raise ValueError("Provide only one of prompt_snapshot_id or graph_snapshot_id.")

        url = f"{self.backend_url}/graphgen/graph/completions"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "job_id": self.job_id,
            "input": input_data,
        }
        if model:
            payload["model"] = model
        snapshot_id = graph_snapshot_id or prompt_snapshot_id
        if snapshot_id:
            payload["prompt_snapshot_id"] = snapshot_id

        resp = http_post(url, headers=headers, json_body=payload, timeout=60.0)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Inference failed: {resp.status_code} - {resp.text[:500]}"
            )

        return resp.json()

    def run_inference_output(
        self,
        input_data: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt_snapshot_id: Optional[str] = None,
        graph_snapshot_id: Optional[str] = None,
    ) -> Any:
        """Convenience wrapper returning only the model output."""
        result = self.run_inference(
            input_data,
            model=model,
            prompt_snapshot_id=prompt_snapshot_id,
            graph_snapshot_id=graph_snapshot_id,
        )
        if isinstance(result, dict):
            return result.get("output")
        return None

    def get_graph_record(
        self,
        *,
        prompt_snapshot_id: Optional[str] = None,
        graph_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the optimized graph record (snapshot) for a completed job.

        Note: for graph-first jobs, this record is **redacted** and never
        includes proprietary YAML/spec. Use `download_graph_txt()` for the
        public export.

        Args:
            prompt_snapshot_id: Legacy alias for selecting a specific snapshot.
            graph_snapshot_id: Specific GraphSnapshot to use (default: best).

        Returns:
            Graph record dictionary containing:
            - job_id: The job ID
            - snapshot_id: The snapshot ID used
            - prompt: Extracted prompt text (legacy single-node only; may be empty)
            - graph: Public graph record payload (e.g., export metadata)
            - model: Model used for this graph (optional)

        Raises:
            RuntimeError: If job hasn't been submitted
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        if prompt_snapshot_id and graph_snapshot_id:
            raise ValueError("Provide only one of prompt_snapshot_id or graph_snapshot_id.")

        url = f"{self.backend_url}/graphgen/graph/record"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "job_id": self.job_id,
        }
        snapshot_id = graph_snapshot_id or prompt_snapshot_id
        if snapshot_id:
            payload["prompt_snapshot_id"] = snapshot_id

        resp = http_post(url, headers=headers, json_body=payload, timeout=30.0)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to get graph record: {resp.status_code} - {resp.text[:500]}"
            )

        return resp.json()


__all__ = [
    "GraphGenJob",
    "GraphGenJobResult",
    "GraphGenSubmitResult",
]
