"""GraphGen (Automated Design of Agentic Systems) data models.

This module provides Pydantic models for defining GraphGen datasets and job configurations.
GraphGen is a simplified "Workflows API" for prompt optimization that wraps GEPA with
auto-generated task apps and built-in judge configurations.

Example:
    from synth_ai.sdk.api.train.graphgen_models import (
        GraphGenTaskSet,
        GraphGenTask,
        GraphGenGoldOutput,
        GraphGenRubric,
        GraphGenJobConfig,
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
        judge_config=GraphGenJudgeConfig(mode="rubric"),
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


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
    select_output: Optional[Union[str, List[str]]] = Field(
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

    @field_validator("select_output", mode="before")
    @classmethod
    def validate_select_output(
        cls, v: Any
    ) -> Optional[Union[str, List[str]]]:
        """Validate select_output is a string or list of strings."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if all(isinstance(item, str) for item in v):
                return v
            raise ValueError("select_output list must contain only strings")
        raise ValueError(
            f"select_output must be a string or list of strings, got {type(v).__name__}"
        )

    @field_validator("output_config", mode="before")
    @classmethod
    def validate_output_config(cls, v: Any) -> Optional[OutputConfig]:
        """Convert dict to OutputConfig for backward compatibility."""
        if v is None:
            return None
        if isinstance(v, OutputConfig):
            return v
        if isinstance(v, dict):
            return OutputConfig.model_validate(v)
        raise ValueError(
            f"output_config must be a dict or OutputConfig, got {type(v).__name__}"
        )


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
    input: Dict[str, Any] = Field(
        ..., description="Arbitrary JSON input for the task"
    )
    rubric: Optional[GraphGenRubric] = Field(
        default=None, description="Task-specific rubric (merged with default_rubric)"
    )


class GraphGenGoldOutput(BaseModel):
    """A gold/reference output.

    Can be linked to a specific task via task_id, or standalone (for reference examples).
    Standalone gold outputs (no task_id) are used as reference pool for contrastive judging.
    """

    output: Dict[str, Any] = Field(
        ..., description="The gold/reference output (arbitrary JSON)"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="ID of the task this gold output belongs to (None = standalone reference)",
    )
    note: Optional[str] = Field(
        default=None, description="Optional note about this gold output"
    )


# Improvement 4: Define supported providers as a Literal type
JudgeProviderType = Literal["groq", "openai", "google", "anthropic"]


class GraphGenJudgeConfig(BaseModel):
    """Configuration for the judge used during optimization."""

    mode: Literal["rubric", "contrastive", "gold_examples"] = Field(
        default="rubric",
        description=(
            "Judge mode: "
            "'rubric' = evaluate against criteria, "
            "'contrastive' = compare to gold output, "
            "'gold_examples' = use gold examples as few-shot context"
        ),
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model to use for judging",
    )
    # Improvement 4: Changed from str to Literal type for better type safety
    provider: JudgeProviderType = Field(
        default="groq",
        description="Provider for judge model (groq, openai, google, anthropic)",
    )


class GraphGenTaskSet(BaseModel):
    """The complete GraphGen dataset format.

    Contains tasks with arbitrary JSON inputs, gold outputs (optionally linked to tasks),
    rubrics (task-specific and/or default), and judge configuration.

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
    tasks: List[GraphGenTask] = Field(
        ..., min_length=1, description="List of tasks to evaluate"
    )
    gold_outputs: List[GraphGenGoldOutput] = Field(
        default_factory=list,
        description="Gold/reference outputs (linked to tasks or standalone)",
    )
    default_rubric: Optional[GraphGenRubric] = Field(
        default=None,
        description="Default rubric applied to all tasks (merged with task-specific rubrics)",
    )
    judge_config: GraphGenJudgeConfig = Field(
        default_factory=GraphGenJudgeConfig,
        description="Configuration for the judge",
    )
    # Optional schemas (also accepted at top-level for backward/forward compatibility).
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for task inputs / initial_state"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for expected graph output / final_state"
    )
    # Improvement 3: Changed from Optional[Any] to Optional[Union[str, List[str]]]
    select_output: Optional[Union[str, List[str]]] = Field(
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

    @field_validator("tasks")
    @classmethod
    def validate_unique_task_ids(cls, v: List[GraphGenTask]) -> List[GraphGenTask]:
        """Ensure all task IDs are unique."""
        ids = [task.id for task in v]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate task IDs found: {set(duplicates)}")
        return v

    # Improvement 2: Added ValidationInfo type hint
    @field_validator("gold_outputs")
    @classmethod
    def validate_gold_output_task_ids(
        cls, v: List[GraphGenGoldOutput], info: ValidationInfo
    ) -> List[GraphGenGoldOutput]:
        """Ensure gold output task_ids reference valid tasks.

        Args:
            v: The list of gold outputs being validated.
            info: Pydantic ValidationInfo providing access to other fields via info.data.

        Returns:
            The validated list of gold outputs.

        Raises:
            ValueError: If a gold output references a non-existent task ID.
        """
        tasks = info.data.get("tasks", [])
        if tasks:
            # Improvement 8: Handle both GraphGenTask objects and raw dicts during validation
            valid_task_ids: set[str] = set()
            for task in tasks:
                if isinstance(task, GraphGenTask):
                    valid_task_ids.add(task.id)
                elif isinstance(task, dict):
                    # During validation, tasks might still be raw dicts
                    task_id = task.get("id")
                    if task_id:
                        valid_task_ids.add(task_id)

            for gold in v:
                if gold.task_id and gold.task_id not in valid_task_ids:
                    raise ValueError(
                        f"Gold output references invalid task_id: {gold.task_id}"
                    )
        return v

    # Improvement 3: Validator for select_output type
    @field_validator("select_output", mode="before")
    @classmethod
    def validate_select_output(
        cls, v: Any
    ) -> Optional[Union[str, List[str]]]:
        """Validate select_output is a string or list of strings."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if all(isinstance(item, str) for item in v):
                return v
            raise ValueError("select_output list must contain only strings")
        raise ValueError(
            f"select_output must be a string or list of strings, got {type(v).__name__}"
        )

    # Improvement 1: Validator for backward-compatible OutputConfig
    @field_validator("output_config", mode="before")
    @classmethod
    def validate_output_config(cls, v: Any) -> Optional[OutputConfig]:
        """Convert dict to OutputConfig for backward compatibility."""
        if v is None:
            return None
        if isinstance(v, OutputConfig):
            return v
        if isinstance(v, dict):
            return OutputConfig.model_validate(v)
        raise ValueError(
            f"output_config must be a dict or OutputConfig, got {type(v).__name__}"
        )

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
        """Get gold outputs not linked to any task (reference pool for contrastive judge)."""
        return [gold for gold in self.gold_outputs if gold.task_id is None]


# Supported policy models (same as prompt opt)
SUPPORTED_POLICY_MODELS = {
    # Groq (fast, free tier friendly)
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    # OpenAI
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    # OpenAI - GPT-4.1 series
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # Gemini
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash-image",  # Nano Banana - image generation model
    # Claude
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
}

# Supported judge models
SUPPORTED_JUDGE_MODELS = {
    # Groq (fast, cheap)
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    # OpenAI (higher quality)
    "gpt-4o-mini",
    "gpt-4o",
}

# Default models
DEFAULT_POLICY_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"
DEFAULT_JUDGE_PROVIDER = "groq"


class GraphGenJobConfig(BaseModel):
    """Configuration for an GraphGen optimization job.

    Example:
        config = GraphGenJobConfig(
            policy_model="gpt-4o-mini",
            rollout_budget=100,
            proposer_effort="medium",
        )
    """

    # Policy model (what the prompt runs on)
    policy_model: str = Field(
        default=DEFAULT_POLICY_MODEL,
        description="Model to use for policy inference",
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

    # Proposer settings (controls prompt mutation quality/cost)
    proposer_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Proposer effort level (affects mutation quality and cost)",
    )

    # Judge settings (if not specified in dataset)
    judge_model: Optional[str] = Field(
        default=None,
        description="Override judge model from dataset",
    )
    judge_provider: Optional[str] = Field(
        default=None,
        description="Override judge provider from dataset",
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

    def get_policy_provider(self) -> str:
        """Get the policy provider (auto-detect if not specified)."""
        if self.policy_provider:
            return self.policy_provider
        return _detect_provider(self.policy_model)


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    model_lower = model.lower()
    if model_lower.startswith("gpt-") or model_lower.startswith("o1"):
        return "openai"
    elif model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("claude"):
        return "anthropic"
    elif "llama" in model_lower or "mixtral" in model_lower:
        return "groq"
    else:
        # Default to OpenAI for unknown models
        return "openai"


def parse_graphgen_taskset(data: Dict[str, Any]) -> GraphGenTaskSet:
    """Parse a dictionary into an GraphGenTaskSet.

    Args:
        data: Dictionary containing the taskset data (from JSON)

    Returns:
        Validated GraphGenTaskSet

    Raises:
        ValueError: If validation fails
    """
    try:
        return GraphGenTaskSet.model_validate(data)
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return parse_graphgen_taskset(data)


# GraphGen aliases (preferred names)
GraphGenTaskSet = GraphGenTaskSet
GraphGenTaskSetMetadata = GraphGenTaskSetMetadata
GraphGenTask = GraphGenTask
GraphGenGoldOutput = GraphGenGoldOutput
GraphGenRubric = GraphGenRubric
GraphGenRubricCriterion = GraphGenRubricCriterion
GraphGenRubricOutcome = GraphGenRubricOutcome
GraphGenRubricEvents = GraphGenRubricEvents
GraphGenJudgeConfig = GraphGenJudgeConfig
GraphGenJobConfig = GraphGenJobConfig
parse_graphgen_taskset = parse_graphgen_taskset
load_graphgen_taskset = load_graphgen_taskset

__all__ = [
    # Core types (new)
    "OutputConfig",
    "JudgeProviderType",
    # GraphGen names (preferred)
    "GraphGenTaskSet",
    "GraphGenTaskSetMetadata",
    "GraphGenTask",
    "GraphGenGoldOutput",
    "GraphGenRubric",
    "GraphGenRubricCriterion",
    "GraphGenRubricOutcome",
    "GraphGenRubricEvents",
    "GraphGenJudgeConfig",
    "GraphGenJobConfig",
    "parse_graphgen_taskset",
    "load_graphgen_taskset",
    # Constants
    "SUPPORTED_POLICY_MODELS",
    "SUPPORTED_JUDGE_MODELS",
    "DEFAULT_POLICY_MODEL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_PROVIDER",
]
