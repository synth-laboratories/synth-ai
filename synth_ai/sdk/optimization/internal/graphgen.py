"""Internal graph evolve implementation.

Public API: Use `synth_ai.sdk.optimization.GraphOptimizationJob.from_dataset()` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from .graph_evolve_builder import (
    build_graph_evolve_config,
    build_placeholder_dataset,
    normalize_policy_models,
    parse_graph_evolve_dataset,
    resolve_graph_evolve_credentials,
)
from .graph_evolve_payloads import build_graph_record_payload, build_inference_payload
from .graph_evolve_service import (
    cancel_graph_evolve_job,
    download_graph_evolve_graph_txt,
    download_graph_evolve_prompt,
    get_graph_evolve_events,
    get_graph_evolve_graph_record,
    get_graph_evolve_metrics,
    get_graph_evolve_status,
    query_graph_evolve_workflow_state,
    run_graph_evolve_inference,
    start_graph_evolve_job,
    submit_graph_evolve_job,
)
from .graphgen_models import GraphGenJobConfig as GraphEvolveJobConfig
from .graphgen_models import GraphGenTaskSet as GraphEvolveTaskSet
from .utils import ensure_api_base, run_sync


@dataclass
class GraphEvolveJobResult:
    """Result from a Graph Evolve job."""

    graph_evolve_job_id: str
    status: str
    best_score: Optional[float] = None
    best_snapshot_id: Optional[str] = None
    error: Optional[str] = None
    dataset_name: Optional[str] = None
    task_count: Optional[int] = None
    legacy_graphgen_job_id: Optional[str] = None


@dataclass
class GraphEvolveSubmitResult:
    """Result from submitting a Graph Evolve job."""

    graph_evolve_job_id: str
    status: str
    dataset_name: str
    task_count: int
    rollout_budget: int
    policy_models: List[str]
    judge_mode: str
    legacy_graphgen_job_id: Optional[str] = None


class GraphEvolveJob:
    """High-level SDK class for running Graph Evolve workflow optimization jobs.

    Graph Evolve (Automated Design of Agentic Systems) provides a simplified API for
    graph/workflow optimization that doesn't require users to manage task apps.

    Key differences from PromptLearningJob:
    - Uses JSON dataset format (GraphEvolveTaskSet) instead of TOML configs
    - No task app management required - Graph Evolve builds it internally
    - Built-in judge modes (rubric, contrastive, gold_examples)
    - Graph-first: trains multi-node workflows by default (Graph-GEPA)
    - Public graph downloads are redacted `.txt` exports only
    - Simpler configuration with sensible defaults

    Example:
        >>> from synth_ai.sdk.optimization.internal.graphgen import GraphEvolveJob
        >>>
        >>> # Create job from dataset file
        >>> job = GraphEvolveJob.from_dataset(
        ...     dataset="my_tasks.json",
        ...     policy_models="gpt-4o-mini",
        ...     rollout_budget=100,
        ... )
        >>>
        >>> # Submit and stream
        >>> job.submit()
        >>> result = job.stream_until_complete(timeout=3600.0)
        >>> print(f"Best reward: {result.get('best_reward')}")
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
        dataset: GraphEvolveTaskSet,
        config: GraphEvolveJobConfig,
        backend_url: str,
        api_key: str,
        auto_start: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize an GraphEvolve job.

        Args:
            dataset: The GraphEvolveTaskSet containing tasks and evaluation config
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

        self._graph_evolve_job_id: Optional[str] = None
        self._legacy_graphgen_job_id: Optional[str] = None
        self._submit_result: Optional[GraphEvolveSubmitResult] = None

    @classmethod
    def from_dataset(
        cls,
        dataset: str | Path | Dict[str, Any] | GraphEvolveTaskSet,
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
    ) -> GraphEvolveJob:
        """Create an GraphEvolve job from a dataset.

        Args:
            dataset: Dataset as file path, dict, or GraphEvolveTaskSet object
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
            graph_type: Type of graph to train - "policy" (default), "verifier", or "rlm"
            initial_graph_id: Preset graph ID to optimize. Required.
                Graph Evolve now runs prompt-only GEPA on a fixed graph.
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            auto_start: Whether to start the job immediately
            metadata: Additional metadata for the job

        Returns:
            GraphEvolveJob instance

        Example:
            >>> # From file with single model
            >>> job = GraphEvolveJob.from_dataset("tasks.json", policy_models="gpt-4o-mini")
            >>>
            >>> # From file with multiple models
            >>> job = GraphEvolveJob.from_dataset("tasks.json", policy_models=["gpt-4o-mini", "gpt-4.1-mini"])
            >>>
            >>> # From dict
            >>> job = GraphEvolveJob.from_dataset({
            ...     "metadata": {"name": "My Tasks"},
            ...     "initial_prompt": "You are helpful.",
            ...     "tasks": [{"id": "t1", "input": {"q": "Hi"}}],
            ... })
            >>>
            >>> # From GraphEvolveTaskSet object
            >>> job = GraphEvolveJob.from_dataset(my_taskset, policy_models=["gpt-4o"])
        """
        parsed_dataset = parse_graph_evolve_dataset(dataset)
        backend_url, api_key = resolve_graph_evolve_credentials(
            backend_url=backend_url,
            api_key=api_key,
        )
        policy_models_list = normalize_policy_models(policy_models)
        config = build_graph_evolve_config(
            policy_models=policy_models_list,
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
    ) -> GraphEvolveJob:
        """Resume an existing GraphEvolve job by ID.

        Args:
            job_id: Graph Evolve job ID ("graph_evolve_*"), legacy GraphGen job ID
                ("graphgen_*"), or underlying GEPA job ID ("pl_*")
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            GraphEvolveJob instance for the existing job
        """
        backend_url, api_key = resolve_graph_evolve_credentials(
            backend_url=backend_url,
            api_key=api_key,
        )

        # Create minimal instance - dataset will be fetched from backend if needed.
        placeholder_dataset = build_placeholder_dataset()

        job = cls(
            dataset=placeholder_dataset,
            config=GraphEvolveJobConfig(
                policy_models=["(resumed)"]
            ),  # Placeholder, will be fetched from backend
            backend_url=backend_url,
            api_key=api_key,
            auto_start=False,
        )

        # Accept Graph Evolve, legacy GraphGen, or GEPA job IDs.
        valid_prefixes = ("graph_evolve_", "graphgen_", "pl_")
        if not any(job_id.startswith(p) for p in valid_prefixes):
            raise ValueError(
                f"Unsupported job ID format: {job_id!r}. Expected one of: {valid_prefixes}"
            )
        if job_id.startswith("graphgen_"):
            job._legacy_graphgen_job_id = job_id
        else:
            job._graph_evolve_job_id = job_id
        return job

    @classmethod
    def from_graph_evolve_job_id(
        cls,
        graph_evolve_job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphEvolveJob:
        """Alias for resuming a Graph Evolve job from a GEPA job ID."""
        return cls.from_job_id(graph_evolve_job_id, backend_url=backend_url, api_key=api_key)

    @property
    def job_id(self) -> Optional[str]:
        """Get the Graph Evolve job ID (None if not yet submitted)."""
        if self._graph_evolve_job_id:
            return self._graph_evolve_job_id
        if self._submit_result and self._submit_result.graph_evolve_job_id:
            return self._submit_result.graph_evolve_job_id
        return self._legacy_graphgen_job_id

    @property
    def graph_evolve_job_id(self) -> Optional[str]:
        """Get the underlying GEPA job ID if known."""
        if self._graph_evolve_job_id and self._graph_evolve_job_id.startswith("pl_"):
            return self._graph_evolve_job_id
        if (
            self._submit_result
            and self._submit_result.graph_evolve_job_id
            and self._submit_result.graph_evolve_job_id.startswith("pl_")
        ):
            return self._submit_result.graph_evolve_job_id
        return None

    @property
    def legacy_graphgen_job_id(self) -> Optional[str]:
        """Get the legacy GraphGen job ID if known."""
        if self._legacy_graphgen_job_id:
            return self._legacy_graphgen_job_id
        if self._submit_result and self._submit_result.legacy_graphgen_job_id:
            return self._submit_result.legacy_graphgen_job_id
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
        # GraphEvolve is graph-first and doesn't really use this, so we use a placeholder or problem_spec
        dataset_dict = self.dataset.model_dump(mode="json", exclude_none=False)
        if "initial_prompt" not in dataset_dict:
            dataset_dict["initial_prompt"] = (
                self.config.problem_spec or "Optimizing prompt graph..."
            )

        # Ensure verifier_config is properly included
        if (
            "verifier_config" not in dataset_dict or dataset_dict.get("verifier_config") is None
        ) and self.dataset.verifier_config:
            # Fallback: create verifier_config from dataset's verifier_config
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

        # Add initial_graph_id if provided
        if self.config.initial_graph_id:
            payload["initial_graph_id"] = self.config.initial_graph_id

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

    def submit(self) -> GraphEvolveSubmitResult:
        """Submit the job to the backend.

        Returns:
            GraphEvolveSubmitResult with job IDs and initial status

        Raises:
            RuntimeError: If job submission fails
        """
        from .graphgen_validators import validate_graphgen_job_config

        if self._graph_evolve_job_id or self._legacy_graphgen_job_id:
            raise RuntimeError(f"Job already submitted: {self.job_id}")

        # Validate config + dataset before expensive API call.
        validate_graphgen_job_config(self.config, self.dataset)

        payload = self._build_payload()

        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Submitting Graph Evolve job to: %s", self.backend_url)

        js = submit_graph_evolve_job(
            backend_url=self.backend_url,
            api_key=self.api_key,
            payload=payload,
        )

        self._graph_evolve_job_id = js.get("graph_evolve_job_id")
        self._legacy_graphgen_job_id = js.get("graphgen_job_id")

        if not self._graph_evolve_job_id and not self._legacy_graphgen_job_id:
            raise RuntimeError("Response missing graph_evolve_job_id")

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

        self._submit_result = GraphEvolveSubmitResult(
            graph_evolve_job_id=self.job_id or "",
            status=js.get("status", "queued"),
            dataset_name=js.get("dataset_name", self.dataset.metadata.name),
            task_count=js.get("task_count", len(self.dataset.tasks)),
            rollout_budget=js.get("rollout_budget", self.config.rollout_budget),
            policy_models=policy_models_response,
            judge_mode=judge_mode,
            legacy_graphgen_job_id=self._legacy_graphgen_job_id,
        )

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

        data = get_graph_evolve_status(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
        )
        gepa_id = data.get("graph_evolve_job_id")
        if gepa_id:
            self._graph_evolve_job_id = gepa_id
        return data

    def start(self) -> Dict[str, Any]:
        """Start a queued GraphEvolve job.

        This is only needed if the job was created with auto_start=False or ended up queued.
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        data = start_graph_evolve_job(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
        )
        if self._submit_result and "status" in data:
            self._submit_result.status = data.get("status", self._submit_result.status)
        return data

    def get_events(self, *, since_seq: int = 0, limit: int = 1000) -> Dict[str, Any]:
        """Fetch events for this GraphEvolve job.

        Returns backend envelope: {"events": [...], "has_more": bool, "next_seq": int}.
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return get_graph_evolve_events(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
            since_seq=since_seq,
            limit=limit,
        )

    def get_metrics(
        self,
        *,
        name: Optional[str] = None,
        after_step: Optional[int] = None,
        limit: int = 500,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch metrics for this Graph Evolve job.

        Mirrors GET /api/graph-evolve/jobs/{job_id}/metrics.
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
        return get_graph_evolve_metrics(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
            query_string=qs,
        )

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Stream job events until completion using SSE (async)."""
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        from .graph_evolve_streaming import build_graph_evolve_streamer

        streamer = build_graph_evolve_streamer(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
            handlers=handlers,
            interval=interval,
            timeout=timeout,
        )

        final_status = await streamer.stream_until_terminal()

        if on_event and isinstance(final_status, dict):
            on_event(final_status)

        return final_status

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Stream job events until completion using SSE.

        Note: on_event is invoked once with the final status payload; use handlers
        to receive per-event callbacks.
        """
        return run_sync(
            self.stream_until_complete_async(
                timeout=timeout,
                interval=interval,
                handlers=handlers,
                on_event=on_event,
            ),
            label="stream_until_complete() (use stream_until_complete_async in async contexts)",
        )

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

        data = download_graph_evolve_prompt(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
        )
        return data.get("prompt", "")

    def download_graph_txt(self) -> str:
        """Download a PUBLIC (redacted) graph export for a completed job.

        Graph-first GraphEvolve jobs produce multi-node graphs. The internal graph
        YAML/spec is proprietary and never exposed. This helper downloads the
        `.txt` export from:
            GET /api/graph-evolve/jobs/{job_id}/graph.txt
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return download_graph_evolve_graph_txt(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
        )

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

        payload = build_inference_payload(
            job_id=self.job_id,
            input_data=input_data,
            model=model,
            prompt_snapshot_id=prompt_snapshot_id,
            graph_snapshot_id=graph_snapshot_id,
        )

        return run_graph_evolve_inference(
            backend_url=self.backend_url,
            api_key=self.api_key,
            payload=payload,
        )

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

        payload = build_graph_record_payload(
            job_id=self.job_id,
            prompt_snapshot_id=prompt_snapshot_id,
            graph_snapshot_id=graph_snapshot_id,
        )

        return get_graph_evolve_graph_record(
            backend_url=self.backend_url,
            api_key=self.api_key,
            payload=payload,
        )

    def cancel(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a running GraphEvolve job.

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
            HTTPStatusError: If the cancellation request fails

        Example:
            >>> job.submit()
            >>> # Later...
            >>> result = job.cancel(reason="No longer needed")
            >>> print(result["message"])
            "Temporal workflow cancelled successfully."
        """
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        payload: Dict[str, Any] = {}
        if reason:
            payload["reason"] = reason
        return cancel_graph_evolve_job(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
            payload=payload,
        )

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
                - phase: Current graph optimization phase
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
        if not self.job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return query_graph_evolve_workflow_state(
            backend_url=self.backend_url,
            api_key=self.api_key,
            job_id=self.job_id,
        )


# Legacy aliases (GraphGen naming)
GraphGenJob = GraphEvolveJob
GraphGenJobResult = GraphEvolveJobResult
GraphGenSubmitResult = GraphEvolveSubmitResult


__all__ = [
    "GraphEvolveJob",
    "GraphEvolveJobResult",
    "GraphEvolveSubmitResult",
]
