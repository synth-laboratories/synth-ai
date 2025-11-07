"""Core dataclasses for baseline configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class BaselineTaskRunner:
    """
    Base class for task runners.
    
    Subclasses should implement `run_task` method for class-based approach,
    or you can use standalone async functions for function-based approach.
    """
    
    def __init__(
        self,
        policy_config: Dict[str, Any],
        env_config: Dict[str, Any],
    ):
        """
        Initialize task runner with configuration.
        
        Args:
            policy_config: Policy configuration (model, temperature, etc.)
            env_config: Environment configuration (max_steps, difficulty, etc.)
        """
        self.policy_config = policy_config
        self.env_config = env_config
    
    async def run_task(self, seed: int) -> TaskResult:
        """
        Execute a single task instance.
        
        This method is called for each seed in the selected split.
        
        Args:
            seed: The seed/index for this task instance
        
        Returns:
            TaskResult: Structured result containing success, rewards, metadata, trace
        """
        raise NotImplementedError("Subclasses must implement run_task method")


@dataclass
class DataSplit:
    """Definition of a data split (train/val/test)."""
    
    name: str  # "train", "val", "test"
    seeds: List[int]  # Seed/index values for this split
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional metadata


@dataclass
class TaskResult:
    """Result from a single task execution."""
    
    # Required: Seed/index that was evaluated
    seed: int
    
    # Required: Did the task complete successfully?
    success: bool
    
    # Required: Outcome reward for the episode
    outcome_reward: float
    
    # Optional: Event rewards (step-level)
    event_rewards: List[Dict[str, Any]] = field(default_factory=list)
    
    # Optional: Total steps/turns taken
    total_steps: int = 0
    
    # Optional: Metadata (achievements, completion info, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: Error information if success=False
    error: Optional[str] = None
    
    # Optional: v3 trace (SessionTrace dict)
    trace: Optional[Dict[str, Any]] = None


# Type alias for task runner (can be class or function)
TaskRunnerType = (
    type[BaselineTaskRunner]
    | Callable[[int, dict[str, Any], dict[str, Any]], Any]  # Function signature
)

# Type alias for result aggregator (can be class or function)
AggregatorType = (
    type[Any]  # Class with aggregate() method
    | Callable[[list[TaskResult]], dict[str, Any]]  # Function signature
)


@dataclass
class BaselineConfig:
    """Configuration for a baseline file.
    
    A baseline file defines how to evaluate a task without requiring
    a deployed task app. It provides self-contained evaluation logic
    with first-class support for train/val/test splits.
    
    Supports both class-based and function-based task runners:
    - Class-based: Pass a class that inherits from BaselineTaskRunner
    - Function-based: Pass an async function with signature:
      async def task_runner(seed: int, policy_config: Dict[str, Any], 
                           env_config: Dict[str, Any]) -> TaskResult
    """
    
    # Required: Unique identifier for this baseline config
    baseline_id: str
    
    # Required: Human-readable name
    name: str
    
    # Required: Task runner (class or function)
    # Class-based: Pass a class inheriting from BaselineTaskRunner
    #   The class will be instantiated with policy_config and env_config,
    #   and run_task(seed) will be called for each seed.
    # Function-based: Pass an async function with signature:
    #   async def task_runner(seed: int, policy_config: Dict[str, Any], 
    #                        env_config: Dict[str, Any]) -> TaskResult
    task_runner: TaskRunnerType
    
    # Required: Data splits (train/val/test)
    splits: Dict[str, DataSplit]
    
    # Optional: Description for documentation
    description: str = ""
    
    # Optional: Default policy configuration
    default_policy_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: Default environment configuration
    default_env_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: Metadata for filtering/organization
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: Tags for filtering and discovery
    tags: List[str] = field(default_factory=list)
    
    # Optional: Custom result aggregator (class or function)
    # Class-based: Pass a class with aggregate(results: List[TaskResult]) method
    #   The class will be instantiated and aggregate() called.
    # Function-based: Pass a function with signature:
    #   def aggregate_results(results: List[TaskResult]) -> Dict[str, Any]
    result_aggregator: Optional[AggregatorType] = None
    
    # Optional: Path to this baseline file (set by discovery)
    _source_path: Optional[Path] = None
    
    def matches_tag(self, tag: str) -> bool:
        """Check if baseline matches a tag (case-insensitive)."""
        return tag.lower() in [t.lower() for t in self.tags]
    
    def matches_metadata(self, key: str, value: Any) -> bool:
        """Check if baseline metadata matches key-value pair."""
        return self.metadata.get(key) == value


@dataclass
class BaselineResults:
    """Aggregate results from a baseline evaluation."""
    
    # Configuration that was used
    config: BaselineConfig
    
    # Split that was evaluated
    split_name: str
    
    # Per-seed results
    results: List[TaskResult]
    
    # Aggregate metrics
    aggregate_metrics: Dict[str, Any]
    
    # Execution metadata
    execution_time_seconds: float
    model_name: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "baseline_id": self.config.baseline_id,
            "name": self.config.name,
            "split": self.split_name,
            "model": self.model_name,
            "timestamp": self.timestamp,
            "execution_time_seconds": self.execution_time_seconds,
            "aggregate_metrics": self.aggregate_metrics,
            "results": [
                {
                    "seed": r.seed,
                    "success": r.success,
                    "outcome_reward": r.outcome_reward,
                    "total_steps": r.total_steps,
                    "metadata": r.metadata,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

