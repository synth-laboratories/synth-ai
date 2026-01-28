"""GraphGen (Graph Opt) data models.

This module provides Pydantic models for defining GraphGen datasets and job configurations.
GraphGen is a simplified "Workflows API" for prompt optimization that wraps GEPA with
auto-generated task apps and built-in verifier configurations.

Example:
    from synth_ai.sdk.optimization.internal.graphgen_models import (
        GraphGenTaskSet,
        GraphGenTask,
        GraphGenGoldOutput,
        GraphGenRubric,
        GraphGenJobConfig,
        GraphGenVerifierConfig,
    )

    # Create a dataset
    dataset = GraphGenTaskSet(
        metadata=GraphGenTaskSetMetadata(name="My Dataset"),
        tasks=[
            GraphGenTask(id="task1", input={"question": "What is 2+2?"}),
            GraphGenTask(id="task2", input={"question": "What is the capital of France?"}),
        ],
        gold_outputs=[
            GraphGenGoldOutput(output={"answer": "4"}, task_id="task1"),
            GraphGenGoldOutput(output={"answer": "Paris"}, task_id="task2"),
        ],
        verifier_config=GraphGenVerifierConfig(mode="rubric"),
    )
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_ai.data.enums import GraphType, RewardSource, RewardType, VerifierMode

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.graphgen_models.") from exc


def _load_graph_opt_config_fallback() -> Dict[str, Any]:
    asset = (
        Path(__file__).resolve().parents[4] / "synth_ai_core" / "assets" / "supported_models.json"
    )
    if asset.exists():
        data = json.loads(asset.read_text())
        return data.get("graph_opt", {})
    return {}


def _require_rust() -> Any | None:
    if synth_ai_py is None:
        return None
    required = (
        "graph_opt_supported_models",
        "validate_graphgen_taskset",
        "parse_graphgen_taskset",
        "load_graphgen_taskset",
        "detect_model_provider",
    )
    missing = [name for name in required if not hasattr(synth_ai_py, name)]
    if missing:
        return None
    return synth_ai_py


# =============================================================================
# Output Configuration (Improvement 1)
# =============================================================================


class OutputConfig(BaseModel):
    """Configuration for graph output extraction + validation.

    This model defines how graph outputs should be extracted and validated.
    It supports JSON Schema validation, multiple output formats, and
    configurable extraction paths.

    Example:
        config = OutputConfig(
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            format="json",
            strict=True,
            extract_from=["response", "output"],
        )

    Attributes:
        schema_: JSON Schema (draft-07) for output validation. Use alias "schema" in JSON.
        format: Expected output format - "json", "text", "tool_calls", or "image".
        strict: If True, validation failures fail the run; if False, log warnings and continue.
        extract_from: Ordered list of dot-paths/keys to try when extracting output from final_state.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="schema",
        description="JSON Schema (draft-07) for output validation",
    )
    format: Literal["json", "text", "tool_calls", "image"] = Field(
        default="json",
        description="Expected output format.",
    )
    strict: bool = Field(
        default=True,
        description="If true, validation failures fail the run; if false, log warnings and continue.",
    )
    extract_from: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ordered list of dot-paths/keys to try when extracting output from a final_state dict. "
            "If omitted/empty, uses the entire provided value."
        ),
    )


class GraphGenTaskSetMetadata(BaseModel):
    """Metadata about the dataset."""

    name: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    version: Optional[str] = "1.0"
    # Optional schemas (graph-first).
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for task inputs / initial_state"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for expected graph output / final_state"
    )
    # Improvement 3: Changed from Optional[Any] to Optional[Union[str, List[str]]]
    select_output: Optional[str | List[str]] = Field(
        default=None,
        description=(
            "Optional selector for the public output model. "
            "Can be a string (single key) or list of keys to extract from final_state."
        ),
    )
    # Improvement 1: Added typed OutputConfig instead of Dict[str, Any]
    output_config: Optional[OutputConfig] = Field(
        default=None,
        description=(
            "Configuration for graph output extraction and validation. "
            "Defines JSON Schema, output format, and extraction paths."
        ),
    )

    pass


class GraphGenRubricCriterion(BaseModel):
    """A single rubric criterion for evaluation."""

    name: str
    description: str
    expected_answer: Optional[str] = None
    weight: float = 1.0


class GraphGenRubricOutcome(BaseModel):
    """Outcome-level rubric (evaluates final output)."""

    criteria: List[GraphGenRubricCriterion] = Field(default_factory=list)


class GraphGenRubricEvents(BaseModel):
    """Event-level rubric (evaluates intermediate steps)."""

    criteria: List[GraphGenRubricCriterion] = Field(default_factory=list)


class GraphGenRubric(BaseModel):
    """Rubric for evaluating task outputs."""

    outcome: Optional[GraphGenRubricOutcome] = None
    events: Optional[GraphGenRubricEvents] = None


class GraphGenTask(BaseModel):
    """A single task in the dataset.

    Tasks have arbitrary JSON inputs and optional task-specific rubrics.
    Gold outputs are stored separately and linked via task_id.
    """

    id: str
    input: Dict[str, Any] = Field(..., description="Arbitrary JSON input for the task")
    rubric: Optional[GraphGenRubric] = Field(
        default=None, description="Task-specific rubric (merged with default_rubric)"
    )


class GraphGenGoldOutput(BaseModel):
    """A gold/reference output.

    Can be linked to a specific task via task_id, or standalone (for reference examples).
    Standalone gold outputs (no task_id) are used as reference pool for contrastive verification.
    """

    output: Dict[str, Any] = Field(..., description="The gold/reference output (arbitrary JSON)")
    task_id: Optional[str] = Field(
        default=None,
        description="ID of the task this gold output belongs to (None = standalone reference)",
    )
    note: Optional[str] = Field(default=None, description="Optional note about this gold output")


# Improvement 4: Define supported providers as a Literal type
VerifierProviderType = Literal["groq", "openai", "google", "anthropic"]


class GraphGenVerifierConfig(BaseModel):
    """Configuration for the verifier used during optimization."""

    mode: VerifierMode = Field(
        default=VerifierMode.RUBRIC,
        description=(
            "Verifier mode: "
            "'rubric' = evaluate against criteria, "
            "'contrastive' = compare to gold output, "
            "'gold_examples' = use gold examples as few-shot context"
        ),
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model to use for verification",
    )
    # Improvement 4: Changed from str to Literal type for better type safety
    provider: VerifierProviderType = Field(
        default="groq",
        description="Provider for verifier model (groq, openai, google, anthropic)",
    )
    scoring_guidelines: Optional[str] = Field(
        default=None,
        description=(
            "Optional scoring guidelines for the verifier. "
            "If not provided, defaults to BINARY scoring (1.0 = exact match, 0.0 = wrong). "
            "Set this to allow partial credit, e.g., 'Allow partial credit for semantically similar answers.'"
        ),
    )


class GraphGenTaskSet(BaseModel):
    """The complete GraphGen dataset format.

    Contains tasks with arbitrary JSON inputs, gold outputs (optionally linked to tasks),
    rubrics (task-specific and/or default), and verifier configuration.

    Example:
        dataset = GraphGenTaskSet(
            metadata=GraphGenTaskSetMetadata(name="QA Dataset"),
            tasks=[
                GraphGenTask(id="q1", input={"question": "What is 2+2?"}),
            ],
            gold_outputs=[
                GraphGenGoldOutput(output={"answer": "4"}, task_id="q1"),
            ],
        )
    """

    version: str = "1.0"
    metadata: GraphGenTaskSetMetadata
    tasks: List[GraphGenTask] = Field(..., min_length=1, description="List of tasks to evaluate")
    gold_outputs: List[GraphGenGoldOutput] = Field(
        default_factory=list,
        description="Gold/reference outputs (linked to tasks or standalone)",
    )
    default_rubric: Optional[GraphGenRubric] = Field(
        default=None,
        description="Default rubric applied to all tasks (merged with task-specific rubrics)",
    )
    verifier_config: GraphGenVerifierConfig = Field(
        default_factory=GraphGenVerifierConfig,
        description="Configuration for the verifier",
    )
    # Optional schemas (also accepted at top-level for backward/forward compatibility).
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for task inputs / initial_state"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for expected graph output / final_state"
    )
    # Improvement 3: Changed from Optional[Any] to Optional[Union[str, List[str]]]
    select_output: Optional[str | List[str]] = Field(
        default=None,
        description=(
            "Optional selector for the public output model. "
            "Can be a string (single key) or list of keys to extract from final_state."
        ),
    )
    # Improvement 1: Added typed OutputConfig instead of Dict[str, Any]
    output_config: Optional[OutputConfig] = Field(
        default=None,
        description=(
            "Configuration for graph output extraction and validation. "
            "Defines JSON Schema, output format, and extraction paths."
        ),
    )

    @model_validator(mode="after")
    def _rust_validate(self) -> GraphGenTaskSet:
        rust = _require_rust()
        if self.output_config is None:
            self.output_config = OutputConfig()
        if rust is not None:
            errors = rust.validate_graphgen_taskset(
                self.model_dump(mode="json", exclude_none=False)
            )
            if errors:
                raise ValueError(f"Invalid GraphGenTaskSet: {errors}")
        return self

    def get_task_by_id(self, task_id: str) -> Optional[GraphGenTask]:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    # Improvement 9: Clarified docstring for get_task_by_index
    def get_task_by_index(self, index: int) -> Optional[GraphGenTask]:
        """Get a task by zero-based index.

        Args:
            index: Zero-based index into tasks list (0 to len(tasks)-1).

        Returns:
            Task at the specified index, or None if index is out of range.

        Note:
            This method does NOT wrap around. For seed-based lookup that wraps
            around, use get_task_by_seed() in GraphGenTaskAppState instead.
        """
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None

    def get_gold_output_for_task(self, task_id: str) -> Optional[GraphGenGoldOutput]:
        """Get the gold output linked to a specific task."""
        for gold in self.gold_outputs:
            if gold.task_id == task_id:
                return gold
        return None

    def get_standalone_gold_outputs(self) -> List[GraphGenGoldOutput]:
        """Get gold outputs not linked to any task (reference pool for contrastive verifier)."""
        return [gold for gold in self.gold_outputs if gold.task_id is None]


# Supported models (single source of truth)
_RUST = _require_rust()
_GRAPH_OPT_CONFIG = (
    _RUST.graph_opt_supported_models() if _RUST is not None else _load_graph_opt_config_fallback()
)

SUPPORTED_POLICY_MODELS = set(_GRAPH_OPT_CONFIG.get("policy_models", []))
SUPPORTED_VERIFIER_MODELS = set(_GRAPH_OPT_CONFIG.get("verifier_models", []))

_DEFAULTS = _GRAPH_OPT_CONFIG.get("defaults", {})
DEFAULT_POLICY_MODEL = _DEFAULTS.get("policy_model", "")
DEFAULT_VERIFIER_MODEL = _DEFAULTS.get("verifier_model", "")
DEFAULT_VERIFIER_PROVIDER = _DEFAULTS.get("verifier_provider", "")


class EventInput(BaseModel):
    """V3-compatible event input for verifier evaluation."""

    model_config = ConfigDict(extra="allow")

    event_id: int = Field(..., description="Unique integer event ID")
    event_type: str = Field(
        ..., description="Type of event (e.g., 'runtime', 'environment', 'llm')"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary event metadata")


class SessionTimeStepInput(BaseModel):
    """V3-compatible session time step input."""

    model_config = ConfigDict(extra="allow")

    step_id: str = Field(..., description="Unique step identifier")
    step_index: int = Field(..., description="Zero-based index of the step")
    turn_number: Optional[int] = Field(default=None, description="Optional turn/round number")
    events: List[EventInput] = Field(..., description="List of events in this timestep")
    step_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional step-level metadata"
    )


class SessionTraceInput(BaseModel):
    """V3-compatible session trace input for verifier evaluation."""

    model_config = ConfigDict(extra="allow")

    session_id: str = Field(..., description="Unique session/trace ID")
    session_time_steps: List[SessionTimeStepInput] = Field(
        ..., description="List of steps in the trajectory"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Global trace metadata")

    @model_validator(mode="before")
    @classmethod
    def _reject_demo_trace_format(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "timesteps" in data and "session_time_steps" not in data:
            raise ValueError(
                "Invalid trace format. Expected V3 SessionTrace with 'session_time_steps', "
                "got demo format with 'timesteps'. Please convert to V3 format."
            )
        return data


class GraphGenGraphVerifierRequest(BaseModel):
    """Request for verifier graph inference."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="GraphGen or GEPA job ID (must be a verifier graph)")
    session_trace: SessionTraceInput = Field(
        ..., description="V3 session trace to evaluate (must include event_ids for reward linking)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for evaluation (rubric, task description, etc.)",
    )
    prompt_snapshot_id: Optional[str] = Field(
        default=None, description="Specific snapshot to use (default: best)"
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_trace_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "session_trace" not in data and "trace" in data:
            data = dict(data)
            data["session_trace"] = data.pop("trace")
        return data


class GraphGenGraphCompletionsModelUsage(BaseModel):
    """Token usage and cost for a single model in a graph completion."""

    model: str = Field(..., description="Model identifier")
    provider: Optional[str] = Field(default=None, description="Provider (openai, anthropic, etc.)")
    elapsed_ms: int = Field(default=0, description="LLM request time in milliseconds")
    prompt_tokens: int = Field(default=0, description="Input tokens used")
    completion_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated cost in USD")


class EventRewardResponse(BaseModel):
    """Event-level reward from verifier evaluation."""

    model_config = ConfigDict(extra="forbid")

    event_id: int = Field(..., description="Integer event id (FK to synth-ai events table)")
    session_id: str = Field(..., description="Session/trace ID this event belongs to")
    reward_value: float = Field(..., description="Reward value for this event")
    reward_type: Optional[RewardType] = Field(
        default=RewardType.EVALUATOR,
        description="Type of reward",
    )
    key: Optional[str] = Field(default=None, description="Optional key/label for the reward")
    turn_number: Optional[int] = Field(
        default=None, description="Turn/timestep number in the trace"
    )
    source: Optional[RewardSource] = Field(
        default=RewardSource.VERIFIER, description="Reward source"
    )
    objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Canonical objectives for this event (e.g., {'reward': 0.9})",
    )
    annotation: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional annotations (feedback, etc.)"
    )

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_reward_source(cls, v: Any) -> Any:
        if v is None or isinstance(v, RewardSource):
            return v
        if isinstance(v, str):
            raw = v.strip().lower()
            if raw in {"verifier", "evaluator"}:
                return RewardSource.VERIFIER
            if raw in {"task_app", "taskapp", "environment", "runner", "human", "env"}:
                return RewardSource.TASK_APP
            if raw == "fused":
                return RewardSource.FUSED
        return v


class OutcomeRewardResponse(BaseModel):
    """Outcome-level reward from verifier evaluation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session/trace ID")
    total_reward: float = Field(..., description="Overall reward/score for the episode (0-1)")
    objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Canonical outcome objectives (e.g., {'reward': 0.9})",
    )
    achievements_count: int = Field(default=0, description="Number of achievements unlocked")
    total_steps: int = Field(default=0, description="Total timesteps in the trace")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata (feedback, etc.)"
    )
    annotation: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional annotations (free-form)"
    )


class GraphGenGraphVerifierResponse(BaseModel):
    """Response from verifier graph inference."""

    started_at: datetime = Field(..., description="When inference request started (UTC)")
    ended_at: datetime = Field(..., description="When inference request completed (UTC)")
    elapsed_ms: int = Field(..., description="Total elapsed time in milliseconds")
    job_id: str = Field(..., description="GEPA job ID")
    snapshot_id: str = Field(..., description="Snapshot ID used for inference")

    # Structured reward outputs (synth-ai compatible)
    event_rewards: List[EventRewardResponse] = Field(
        default_factory=list, description="Per-event rewards"
    )
    outcome_reward: Optional[OutcomeRewardResponse] = Field(
        default=None, description="Episode-level outcome reward"
    )
    outcome_objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Canonical outcome objectives (e.g., {'reward': score})",
    )
    event_objectives: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Per-event objectives aligned with event_rewards ordering",
    )

    raw_output: Optional[Dict[str, Any]] = Field(
        default=None, description="Full raw output from the verifier graph"
    )

    usage: List[GraphGenGraphCompletionsModelUsage] = Field(
        default_factory=list, description="Token usage per model"
    )


class GraphGenJobConfig(BaseModel):
    """Configuration for a GraphGen (Graph Opt) optimization job.

    GraphGen provides a simplified API for training optimized graphs/workflows without
    managing task apps manually. It supports three graph types:
    - **policy**: Standard input-to-output graphs for classification, QA, generation
    - **verifier**: Trace-to-score graphs for verifying/evaluating agent behavior
    - **rlm**: Recursive Language Model graphs for massive contexts via tool-based search

    Example:
        ```python
        from synth_ai.sdk.optimization.internal.graphgen_models import GraphGenJobConfig

        config = GraphGenJobConfig(
            graph_type="policy",
            policy_models=["gpt-4o-mini"],
            rollout_budget=100,
            proposer_effort="medium",
            problem_spec="Classify customer support messages into categories.",
        )
        ```

    Attributes:
        graph_type: Type of graph - "policy", "verifier", or "rlm".
        policy_models: List of models for policy inference (e.g., ["gpt-4o-mini", "claude-3-5-sonnet"]).
            Supports image generation models: OpenAI (gpt-image-1.5, gpt-image-1, gpt-image-1-mini, chatgpt-image-latest)
            and Gemini (gemini-2.5-flash-image, gemini-3-pro-image-preview).
        policy_provider: Provider for policy model (auto-detected if not specified).
        rollout_budget: Total rollouts (evaluations) for optimization. Range: 10-10000.
        proposer_effort: Mutation quality/cost level - "medium" or "high".
            Note: "low" is not allowed (gpt-4.1-mini too weak for graph generation).
        verifier_model: Override verifier model from dataset.
        verifier_provider: Override verifier provider from dataset.
        population_size: GEPA population size. Range: 2-20. Default: 4.
        num_generations: Number of generations (auto-calculated from budget if not specified).
        num_parents: Number of parents for selection. Range: 1-10. Default: 2.
        evaluation_seeds: Specific seeds for evaluation (auto-generated if not specified).
        problem_spec: Detailed problem specification for the graph proposer.
            Include domain info like valid output labels, constraints, format requirements.
        target_llm_calls: Target LLM calls per graph run (1-10). Default: 5.
        configured_tools: Tool bindings for RLM graphs. Required for graph_type="rlm".
        use_byok: BYOK (Bring Your Own Key) mode for rollouts. True = force BYOK (fail if no key),
            False = disable (use Synth credits), None = auto-detect based on org settings.
            When enabled, rollout costs use your own API keys (OpenAI, Anthropic, or Gemini)
            instead of Synth credits. Keys must be configured via /api/v1/byok/keys endpoint.

    Returns:
        After training completes via GraphGenJob, you receive a result dict:
        ```python
        {
            "status": "succeeded",
            "graphgen_job_id": "graphgen_abc123",
            "best_score": 0.89,
            "best_snapshot_id": "snap_xyz789",
            "dataset_name": "My Classification Tasks",
            "task_count": 50,
        }
        ```

    Events:
        During training, you'll receive streaming events via GraphGenJob.stream_until_complete():
        - `graphgen.created` - Job created
        - `graphgen.running` - Training started
        - `graphgen.generation.started` - New generation of candidates started
        - `graphgen.candidate.evaluated` - A candidate graph was evaluated
        - `graphgen.generation.completed` - Generation finished with metrics
        - `graphgen.optimization.completed` - Training finished successfully
        - `graphgen.failed` - Job encountered an error

    See Also:
        - GraphGenJob: High-level SDK class for running jobs
        - GraphGenTaskSet: Dataset format for tasks and gold outputs
        - Training reference: /training/graph-evolve
        - Quickstart: /quickstart/graph-evolve
    """

    # Graph type
    graph_type: GraphType = Field(
        default=GraphType.POLICY,
        description=(
            "Type of graph to train: "
            "'policy' (input->output), "
            "'verifier' (trace->score), "
            "'rlm' (massive context via tool-using RLM-style execution)"
        ),
    )

    # Policy models (what the prompt runs on) - REQUIRED, no default
    policy_models: List[str] = Field(
        ...,
        description="List of models to use for policy inference (allows multiple models)",
    )
    policy_provider: Optional[str] = Field(
        default=None,
        description="Provider for policy model (auto-detected if not specified)",
    )

    # Optimization budget
    rollout_budget: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Total number of rollouts for optimization",
    )
    rollout_max_concurrent: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum parallel rollouts per candidate evaluation",
    )
    rollout_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Timeout per graph rollout execution in seconds",
    )

    # Proposer settings (controls prompt mutation quality/cost)
    proposer_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Proposer effort level (affects mutation quality and cost). "
            "Note: 'low' is not allowed by the backend as gpt-4.1-mini is too weak for graph generation. "
            "Use 'medium' (gpt-4.1) or 'high' (gpt-5.2)."
        ),
    )

    # Verifier settings (if not specified in dataset)
    verifier_model: Optional[str] = Field(
        default=None,
        description="Override verifier model from dataset",
    )
    verifier_provider: Optional[str] = Field(
        default=None,
        description="Override verifier provider from dataset",
    )

    # Advanced settings
    population_size: int = Field(
        default=4,
        ge=2,
        le=20,
        description="Population size for GEPA",
    )
    num_generations: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of generations for GEPA (auto-calculated from budget if not specified)",
    )
    num_parents: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of parents for selection",
    )
    evaluation_seeds: Optional[List[int]] = Field(
        default=None,
        description="Specific seeds to use for evaluation (auto-generated if not specified)",
    )
    initial_graph_id: Optional[str] = Field(
        default=None,
        description="Graph ID to warm-start optimization from. If provided, skips initial graph generation and starts evolution from this graph.",
    )
    problem_spec: Optional[str] = Field(
        default=None,
        description=(
            "Detailed problem specification for the graph proposer. "
            "Include domain-specific information like valid output labels for classification, "
            "constraints, format requirements, or any other info needed to generate correct graphs. "
            "This is combined with the dataset's task_description to form the full proposer context."
        ),
    )
    target_llm_calls: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Target number of LLM calls for the graph (1-10). "
            "Controls max_llm_calls_per_run in the graph_evolve proposer. "
            "If not specified, defaults to 5."
        ),
    )
    configured_tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional job-bound tool allowlist for graph optimization (required for RLM graphs). "
            "Each tool binding should look like: "
            "{'name': 'materialize_context', 'kind': 'rlm_materialize', 'stateful': True} or "
            "{'name': 'local_grep', 'kind': 'rlm_local_grep', 'stateful': False}. "
            "See backend/graphs/tooling/catalog.py for available tool kinds."
        ),
    )

    # BYOK (Bring Your Own Key) - use user's own API keys for rollouts
    use_byok: Optional[bool] = Field(
        default=None,
        description=(
            "BYOK mode: True = force BYOK (fail if no key), "
            "False = disable (use Synth credits), None = auto-detect based on org settings. "
            "When enabled, rollout costs use your own API keys (OpenAI, Anthropic, or Gemini) "
            "instead of Synth credits. Keys must be configured via /api/v1/byok/keys endpoint."
        ),
    )

    def get_policy_provider(self) -> str:
        """Get the policy provider (auto-detect from first policy model if not specified)."""
        if self.policy_provider:
            return self.policy_provider
        # Use first model in list for provider detection
        if self.policy_models:
            return _detect_provider(self.policy_models[0])
        return "openai"  # Default fallback

    @property
    def policy_model(self) -> Optional[str]:
        """Backward-compatible single policy model accessor."""
        return self.policy_models[0] if self.policy_models else None


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    rust = _require_rust()
    if rust is None:
        return "openai"
    return rust.detect_model_provider(model)


def parse_graphgen_taskset(data: Dict[str, Any]) -> GraphGenTaskSet:
    """Parse a dictionary into an GraphGenTaskSet.

    Args:
        data: Dictionary containing the taskset data (from JSON)

    Returns:
        Validated GraphGenTaskSet

    Raises:
        ValueError: If validation fails
    """
    rust = _require_rust()
    if rust is None:
        raise RuntimeError("Rust core graphgen models required; synth_ai_py is unavailable.")
    try:
        parsed = rust.parse_graphgen_taskset(data)
        return GraphGenTaskSet.model_validate(parsed)
    except Exception as e:
        raise ValueError(f"Invalid GraphGenTaskSet format: {e}") from e


def load_graphgen_taskset(path: str | Path) -> GraphGenTaskSet:
    """Load an GraphGenTaskSet from a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Validated GraphGenTaskSet

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    rust = _require_rust()
    if rust is None:
        raise RuntimeError("Rust core graphgen models required; synth_ai_py is unavailable.")
    try:
        parsed = rust.load_graphgen_taskset(str(path))
    except Exception as e:
        raise ValueError(f"Invalid GraphGenTaskSet format: {e}") from e
    return GraphGenTaskSet.model_validate(parsed)


# GraphGen aliases (preferred names)
GraphGenTaskSet = GraphGenTaskSet
GraphGenTaskSetMetadata = GraphGenTaskSetMetadata
GraphGenTask = GraphGenTask
GraphGenGoldOutput = GraphGenGoldOutput
GraphGenRubric = GraphGenRubric
GraphGenRubricCriterion = GraphGenRubricCriterion
GraphGenRubricOutcome = GraphGenRubricOutcome
GraphGenRubricEvents = GraphGenRubricEvents
GraphGenVerifierConfig = GraphGenVerifierConfig
GraphGenJobConfig = GraphGenJobConfig
parse_graphgen_taskset = parse_graphgen_taskset
load_graphgen_taskset = load_graphgen_taskset

__all__ = [
    # Core types (new)
    "OutputConfig",
    "VerifierProviderType",
    # GraphGen names (preferred)
    "GraphGenTaskSet",
    "GraphGenTaskSetMetadata",
    "GraphGenTask",
    "GraphGenGoldOutput",
    "GraphGenRubric",
    "GraphGenRubricCriterion",
    "GraphGenRubricOutcome",
    "GraphGenRubricEvents",
    "GraphGenVerifierConfig",
    "GraphGenJobConfig",
    "parse_graphgen_taskset",
    "load_graphgen_taskset",
    # Constants
    "SUPPORTED_POLICY_MODELS",
    "SUPPORTED_VERIFIER_MODELS",
    "DEFAULT_POLICY_MODEL",
    "DEFAULT_VERIFIER_MODEL",
    "DEFAULT_VERIFIER_PROVIDER",
]
