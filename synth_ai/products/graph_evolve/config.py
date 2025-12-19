"""Configuration for Graph Optimization jobs.

This module provides Pydantic models for loading and validating
graph optimization configuration from TOML files.

Supports multiple algorithms:
- graph_gepa: Grammatical Evolution for graph structure optimization
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

from pydantic import BaseModel, Field, field_validator


class GraphType(str, Enum):
    """What the graph does.

    - POLICY: Maps inputs to outputs (standard LLM pipeline)
    - VERIFIER: Judges/scores existing results
    - RLM: Recursive Language Model - handles massive context (1M+ tokens) by keeping
           it out of prompts and searching via tools. Auto-adds materialize_context,
           local_grep, local_search, query_lm, and codex_exec tools.
    """
    POLICY = "policy"      # Maps inputs to outputs, solves tasks
    VERIFIER = "verifier"  # Judges/scores existing results
    RLM = "rlm"            # Recursive LM - massive context via tool search


class GraphPattern(str, Enum):
    """Architectural patterns for graph structure.

    These patterns can be applied to ANY graph type (policy, verifier, rlm).
    They describe HOW the graph processes data, not WHAT it does.

    Use patterns to guide the proposer toward specific architectures:
    - patterns.required=["rlm"] → MUST use RLM pattern (auto-adds tools)
    - patterns.optional=["rlm", "map_reduce"] → May try either pattern
    - patterns.prefer=["map_reduce"] → Prefer this pattern when viable

    Patterns:
    - RLM: Tool-based search for massive context (1M+ tokens).
           Materializes context to files, uses grep/search tools.
           Good for: RAG, codebase search, document QA.

    - MAP_REDUCE: Parallel processing for variable-length inputs.
           Maps over items in parallel, then reduces/aggregates.
           Good for: scoring events, processing lists, chunking.

    - SINGLE_SHOT: Single LLM call, minimal structure.
           Good for: classification, simple QA.

    - CHAIN_OF_THOUGHT: Multi-step reasoning, sequential nodes.
           Good for: complex reasoning, multi-hop QA.

    - DIGEST_COMBINE: Two-stage: digest in parallel, then combine.
           Good for: verifiers analyzing multiple aspects.
    """
    RLM = "rlm"
    MAP_REDUCE = "map_reduce"
    SINGLE_SHOT = "single_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DIGEST_COMBINE = "digest_combine"


class PatternConfig(BaseModel):
    """Configuration for which architectural patterns the proposer should use.

    Patterns are orthogonal to graph_type - you can have an RLM-pattern verifier
    or a map-reduce-pattern policy. This lets you guide the proposer toward
    specific architectures without changing the fundamental graph type.

    Examples:
        # RLM verifier (MUST use RLM pattern - auto-adds tools & guidance)
        patterns = PatternConfig(required=["rlm"])

        # Let proposer try different patterns
        patterns = PatternConfig(optional=["rlm", "map_reduce", "digest_combine"])

        # Prefer map-reduce but allow alternatives
        patterns = PatternConfig(
            required=[],
            optional=["map_reduce", "digest_combine"],
            prefer=["map_reduce"]
        )

    When RLM pattern is in required/prefer:
        - Tools are auto-added: materialize_context, local_grep, local_search, etc.
        - Proposer receives RLM-specific guidance for tool-based search patterns
    """
    required: List[str] = Field(
        default_factory=list,
        description="Patterns the graph MUST use. Proposer will incorporate all required patterns."
    )
    optional: List[str] = Field(
        default_factory=list,
        description="Patterns the proposer MAY consider. These are suggestions, not requirements."
    )
    prefer: List[str] = Field(
        default_factory=list,
        description="Patterns to prefer when multiple options are viable."
    )

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API request format."""
        return {
            "required": [GraphPattern(p).value for p in self.required],
            "optional": [GraphPattern(p).value for p in self.optional],
            "prefer": [GraphPattern(p).value for p in self.prefer],
        }


class GraphStructure(str, Enum):
    """Structural complexity of the graph."""
    SINGLE_PROMPT = "single_prompt"  # One LLM call, minimal structure
    DAG = "dag"                      # Multiple nodes in sequence, no branching
    CONDITIONAL = "conditional"      # Full graph with conditional branching


class ProposerConfig(BaseModel):
    """Configuration for the LLM proposer."""
    model: str = Field(default="gpt-4.1", description="Model for proposing patches")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class EvolutionConfig(BaseModel):
    """Evolution algorithm parameters."""
    num_generations: int = Field(default=5, ge=1, description="Number of evolution generations")
    children_per_generation: int = Field(default=3, ge=1, description="Children per generation")


class SeedsConfig(BaseModel):
    """Train and validation seed configuration."""
    train: List[int] = Field(default_factory=lambda: list(range(10)))
    validation: List[int] = Field(default_factory=lambda: list(range(100, 105)))


class LimitsConfig(BaseModel):
    """Resource limits for the job."""
    max_spend_usd: float = Field(default=10.0, gt=0)
    timeout_seconds: int = Field(default=3600, gt=0)


class IndifferencePointConfig(BaseModel):
    """Defines trade-off equivalences at a specific anchor point.
    
    Example: At 80% accuracy, 2s latency, $0.50 cost:
      - +2% accuracy change is equivalent to -0.4s latency or -$0.10 cost
      - Differences below 0.5% accuracy are considered noise
    """
    # Anchor point (where trade-offs are defined)
    reward: float = Field(..., description="Anchor reward (e.g., 0.80 for 80%)")
    latency_s: float = Field(..., description="Anchor latency in seconds")
    cost_usd: float = Field(..., description="Anchor cost per seed in USD")
    
    # Trade-off equivalences at this anchor
    reward_delta: float = Field(default=0.02, description="Reward change considered equivalent")
    latency_delta: float = Field(default=0.4, description="Latency change (s) equivalent to reward_delta")
    cost_delta: float = Field(default=0.10, description="Cost change ($) equivalent to reward_delta")
    
    # Noise floors for each objective (differences smaller than these are ignored)
    reward_noise: float = Field(default=0.005, description="Reward diffs below this are noise (e.g., 0.5%)")
    latency_noise: float = Field(default=0.1, description="Latency diffs below this (s) are noise")
    cost_noise: float = Field(default=0.01, description="Cost diffs below this ($) are noise")


class ParetoFloorsConfig(BaseModel):
    """Configuration for multi-objective Pareto comparison.

    Controls:
    1. Which objectives are included in Pareto comparison (enable flags)
    2. Soft floors below which differences are ignored (indifference regions)
    3. Hard ceilings that disqualify candidates entirely (budget constraints)
    """
    # Enable/disable objectives in Pareto comparison
    use_latency: bool = Field(default=True, description="Include latency in Pareto comparison")
    use_cost: bool = Field(default=True, description="Include cost in Pareto comparison")

    # Soft floors: below these, all values are "equally good"
    latency_s: float = Field(default=2.0, description="Don't discriminate on latency below this (s)")
    cost_usd: float = Field(default=0.10, description="Don't discriminate on cost below this ($/seed)")

    # Hard ceilings: disqualify candidates exceeding these limits
    max_latency_s: Optional[float] = Field(default=None, description="Disqualify if mean latency > this")
    max_cost_usd: Optional[float] = Field(default=None, description="Disqualify if mean cost/seed > this")
    min_reward: Optional[float] = Field(default=None, description="Disqualify if mean reward < this")


# ============================================================================
# ADAS Dataset Format Models
# ============================================================================

class TaskInput(BaseModel):
    """A single task/example in an ADAS dataset.

    For POLICY graphs: Contains the problem to solve.
    For VERIFIER graphs: Contains a trace to evaluate.

    Example (Policy - QA):
        {
            "task_id": "q123",
            "input": {
                "question": "What is the capital of France?",
                "context": "Paris is the capital and largest city of France."
            }
        }

    Example (Verifier - Game traces):
        {
            "task_id": "trace_001",
            "input": {
                "game_state": {...},
                "agent_action": "move_north",
                "outcome": "success"
            }
        }
    """
    task_id: Optional[str] = Field(default=None, description="Unique identifier for this task")
    id: Optional[str] = Field(default=None, description="Alternate ID field (task_id preferred)")
    input: Dict[str, Any] = Field(..., description="Input data passed to the graph")

    @field_validator("input", mode="before")
    @classmethod
    def ensure_input_dict(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError(f"Task input must be a dict, got {type(v).__name__}")
        return v

    def get_task_id(self) -> str:
        """Get the task ID, preferring task_id over id."""
        return self.task_id or self.id or "unknown"


class GoldOutput(BaseModel):
    """Ground truth for scoring a task.

    For POLICY graphs: The expected answer/output.
    For VERIFIER graphs: The gold score (0.0-1.0) from human evaluation.

    Example (Policy - QA):
        {
            "task_id": "q123",
            "output": {"answer": "Paris"},
            "score": 1.0  # Optional: gold score for the expected output
        }

    Example (Verifier calibration):
        {
            "task_id": "trace_001",
            "output": {},  # May be empty for verifiers
            "score": 0.75  # Human calibration score
        }
    """
    task_id: Optional[str] = Field(default=None, description="Must match a TaskInput.task_id")
    output: Dict[str, Any] = Field(default_factory=dict, description="Expected output fields")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Gold score (0.0-1.0)")


class ADASDatasetMetadata(BaseModel):
    """Metadata about an ADAS dataset.

    Provides context for graph generation and optimization.
    """
    name: Optional[str] = Field(default=None, description="Dataset name/identifier")
    task_description: Optional[str] = Field(default=None, description="What task this dataset represents")
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="Schema of task inputs")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="Schema of expected outputs")
    domain: Optional[str] = Field(default=None, description="Domain (qa, code, games, etc.)")


class ADASDataset(BaseModel):
    """Complete ADAS dataset format for inline upload.

    This is the schema for the `dataset` field in GraphOptimizationConfig
    when uploading data directly instead of using a pre-registered dataset.

    Example:
        {
            "tasks": [
                {"task_id": "q1", "input": {"question": "What is 2+2?"}},
                {"task_id": "q2", "input": {"question": "What is 3+3?"}}
            ],
            "gold_outputs": [
                {"task_id": "q1", "output": {"answer": "4"}, "score": 1.0},
                {"task_id": "q2", "output": {"answer": "6"}, "score": 1.0}
            ],
            "metadata": {
                "name": "simple_math",
                "task_description": "Answer basic math questions"
            }
        }
    """
    tasks: List[TaskInput] = Field(..., min_length=1, description="List of tasks/examples")
    gold_outputs: List[GoldOutput] = Field(..., min_length=1, description="Ground truth for each task")
    metadata: ADASDatasetMetadata = Field(default_factory=ADASDatasetMetadata)

    @field_validator("tasks", mode="before")
    @classmethod
    def validate_tasks(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError(f"tasks must be a list, got {type(v).__name__}")
        if len(v) == 0:
            raise ValueError("tasks list cannot be empty")
        return v

    @field_validator("gold_outputs", mode="before")
    @classmethod
    def validate_gold_outputs(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError(f"gold_outputs must be a list, got {type(v).__name__}")
        if len(v) == 0:
            raise ValueError("gold_outputs list cannot be empty")
        return v

    def validate_task_ids(self) -> List[str]:
        """Validate that gold_outputs reference valid task IDs.

        Returns list of warnings (non-fatal issues).
        """
        warnings = []
        task_ids = {t.get_task_id() for t in self.tasks}
        gold_task_ids = {g.task_id for g in self.gold_outputs if g.task_id}

        # Check for gold outputs without matching tasks
        orphan_golds = gold_task_ids - task_ids
        if orphan_golds:
            warnings.append(
                f"Gold outputs reference unknown task IDs: {list(orphan_golds)[:5]}"
            )

        # Check for tasks without gold outputs
        missing_golds = task_ids - gold_task_ids
        if missing_golds:
            warnings.append(
                f"Tasks without gold outputs: {list(missing_golds)[:5]}"
            )

        return warnings


class GraphOptimizationConfig(BaseModel):
    """Complete configuration for a graph optimization job.
    
    Example TOML:
        [graph_optimization]
        algorithm = "graph_gepa"
        dataset_name = "hotpotqa"
        graph_type = "policy"
        graph_structure = "dag"
        
        [graph_optimization.evolution]
        num_generations = 5
        children_per_generation = 3
        
        [graph_optimization.proposer]
        model = "gpt-4.1"
        
        [graph_optimization.seeds]
        train = [0, 1, 2, 3, 4]
        validation = [100, 101, 102]
        
        [graph_optimization.limits]
        max_spend_usd = 10.0
    """
    
    # Algorithm selection
    algorithm: str = Field(default="graph_gepa", description="Optimization algorithm (currently: 'graph_gepa')")
    
    # Required
    dataset_name: str = Field(..., description="Dataset to optimize for (e.g., 'hotpotqa')")
    
    # Graph configuration
    graph_type: GraphType = Field(default=GraphType.POLICY)
    graph_structure: GraphStructure = Field(default=GraphStructure.DAG)
    
    # Custom topology guidance (adds detail to graph_structure, doesn't replace it)
    topology_guidance: Optional[str] = Field(
        default=None,
        description="Additional guidance on what kind of graph to build within the chosen structure (e.g., 'Use a single LLM call that reasons and answers in one shot')"
    )

    # Pattern configuration - architectural patterns orthogonal to graph_type
    patterns: Optional[PatternConfig] = Field(
        default=None,
        description=(
            "Configure which architectural patterns the proposer should use/consider. "
            "Patterns are orthogonal to graph_type - you can have an RLM-pattern verifier. "
            "Example: patterns=PatternConfig(required=['rlm']) for RLM-style verifier."
        )
    )

    # Optional warm start from a saved graph in the registry
    initial_graph_id: Optional[str] = Field(
        default=None,
        description="Optional graph_id from the graphs registry to warm-start evolution.",
    )

    # Allowed policy models - which models the generated graph can use
    allowed_policy_models: List[str] = Field(
        default_factory=lambda: ["gpt-4o-mini", "gpt-4o"],
        description="Models the graph is allowed to use in its nodes"
    )
    
    # Nested configs
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    proposer: ProposerConfig = Field(default_factory=ProposerConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    
    # Multi-objective Pareto configuration
    indifference_points: List[IndifferencePointConfig] = Field(
        default_factory=list,
        description="Trade-off equivalences at anchor points"
    )
    pareto_floors: Optional[ParetoFloorsConfig] = Field(
        default=None,
        description="Thresholds below which metric differences are ignored"
    )
    
    # Optional dataset-specific config
    dataset_config: Dict[str, Any] = Field(default_factory=dict)

    # Constraint: max LLM calls per execution
    max_llm_calls_per_run: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum LLM calls allowed per graph execution (e.g., 1, 2, 5).",
    )

    
    # Inline dataset upload (for verifier calibration, custom datasets)
    # Format: {"name": str, "task_description": str, "examples": [...]}
    dataset: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline dataset for upload (ADAS format). If provided, dataset_name is used as identifier."
    )
    
    # Task context for initial graph generation (when dataset doesn't provide it)
    task_description: Optional[str] = Field(default=None, description="Description of the task")
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="Expected input format")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="Expected output format")

    # Problem specification - detailed task info for the graph proposer
    # This should include domain-specific constraints (e.g., valid labels for classification tasks)
    problem_spec: Optional[str] = Field(
        default=None,
        description=(
            "Detailed problem specification for the graph proposer. "
            "Include domain-specific information like valid output labels, constraints, "
            "and any other information needed to generate correct graphs. "
            "If provided, this is combined with task_description for the proposer context."
        )
    )
    
    # Scoring configuration
    scoring_strategy: str = Field(default="rubric", description="Scoring strategy: 'default', 'rubric', 'mae'")
    judge_model: str = Field(default="gpt-4o-mini", description="Model for LLM judge scoring")
    
    @field_validator("graph_type", mode="before")
    @classmethod
    def validate_graph_type(cls, v: Any) -> GraphType:
        if isinstance(v, str):
            return GraphType(v.lower())
        return v
    
    @field_validator("graph_structure", mode="before")
    @classmethod
    def validate_graph_structure(cls, v: Any) -> GraphStructure:
        if isinstance(v, str):
            return GraphStructure(v.lower())
        return v
    
    @classmethod
    def from_toml(cls, path: str | Path) -> "GraphOptimizationConfig":
        """Load configuration from a TOML file.
        
        Args:
            path: Path to the TOML configuration file
            
        Returns:
            Parsed GraphOptimizationConfig
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        # Extract graph_optimization section
        if "graph_optimization" not in data:
            raise ValueError(
                f"Config file must have a [graph_optimization] section. "
                f"Found sections: {list(data.keys())}"
            )
        
        config_data = data["graph_optimization"]
        return cls(**config_data)
    
    def to_request_dict(self) -> Dict[str, Any]:
        """Convert config to API request format.
        
        Returns:
            Dictionary suitable for POST to /graph-gepa/jobs
        """
        request = {
            "dataset_name": self.dataset_name,
            "train_seeds": self.seeds.train,
            "val_seeds": self.seeds.validation,
            "num_generations": self.evolution.num_generations,
            "children_per_generation": self.evolution.children_per_generation,
            "proposer_model": self.proposer.model,
            "graph_type": self.graph_type.value,
            "graph_structure": self.graph_structure.value,
            "allowed_policy_models": self.allowed_policy_models,
            "dataset_config": self.dataset_config,
            "scoring_strategy": self.scoring_strategy,
            "judge_model": self.judge_model,
        }

        if self.max_llm_calls_per_run is not None:
            request["max_llm_calls_per_run"] = int(self.max_llm_calls_per_run)
        
        # Only include topology_guidance if set
        if self.topology_guidance:
            request["topology_guidance"] = self.topology_guidance

        # Include pattern configuration if set
        if self.patterns:
            request["patterns"] = self.patterns.to_api_dict()

        if self.initial_graph_id:
            request["initial_graph_id"] = self.initial_graph_id
        
        # Inline dataset upload (for verifier calibration, custom datasets)
        if self.dataset:
            # Validate dataset structure using Pydantic model
            try:
                validated = ADASDataset(**self.dataset)
                # Check for task ID consistency (non-fatal warnings)
                warnings = validated.validate_task_ids()
                if warnings:
                    import logging
                    logger = logging.getLogger(__name__)
                    for w in warnings:
                        logger.warning(f"[ADASDataset] {w}")
            except Exception as e:
                raise ValueError(
                    f"Invalid ADAS dataset format: {e}\n"
                    f"Expected format: {{'tasks': [...], 'gold_outputs': [...], 'metadata': {{...}}}}\n"
                    f"See ADASDataset model for full schema.\n"
                    f"Got keys: {list(self.dataset.keys())}"
                )
            request["dataset"] = self.dataset
        
        # Task context for initial graph generation
        if self.task_description:
            request["task_description"] = self.task_description
        if self.input_schema:
            request["input_schema"] = self.input_schema
        if self.output_schema:
            request["output_schema"] = self.output_schema
        if self.problem_spec:
            request["problem_spec"] = self.problem_spec

        # Include indifference points for epsilon-Pareto dominance
        if self.indifference_points:
            request["indifference_points"] = [
                p.model_dump() for p in self.indifference_points
            ]
        
        # Include pareto floors for noise reduction
        if self.pareto_floors:
            request["pareto_floors"] = self.pareto_floors.model_dump()
        
        return request
