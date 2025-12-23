"""Type definitions for context learning data structures.

Context Learning optimizes environment setup scripts (pre-flight/post-flight bash)
for terminal/coding agents, similar to how prompt learning optimizes prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentConfig:
    """Environment configuration with pre-flight and post-flight scripts."""
    
    preflight_script: Optional[str] = None
    postflight_script: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentConfig:
        """Create an EnvironmentConfig from a dictionary."""
        return cls(
            preflight_script=data.get("preflight_script"),
            postflight_script=data.get("postflight_script"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        return {
            "preflight_script": self.preflight_script,
            "postflight_script": self.postflight_script,
        }


@dataclass
class AlgorithmConfig:
    """Algorithm configuration for context learning optimization."""
    
    initial_population_size: int = 10
    num_generations: int = 5
    children_per_generation: int = 5
    mutation_llm_model: Optional[str] = None
    mutation_llm_provider: str = "openai"
    policy_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlgorithmConfig:
        """Create an AlgorithmConfig from a dictionary."""
        return cls(
            initial_population_size=data.get("initial_population_size", 10),
            num_generations=data.get("num_generations", 5),
            children_per_generation=data.get("children_per_generation", 5),
            mutation_llm_model=data.get("mutation_llm_model"),
            mutation_llm_provider=data.get("mutation_llm_provider", "openai"),
            policy_config=data.get("policy_config"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: Dict[str, Any] = {
            "initial_population_size": self.initial_population_size,
            "num_generations": self.num_generations,
            "children_per_generation": self.children_per_generation,
            "mutation_llm_provider": self.mutation_llm_provider,
        }
        if self.mutation_llm_model:
            result["mutation_llm_model"] = self.mutation_llm_model
        if self.policy_config:
            result["policy_config"] = self.policy_config
        return result


@dataclass
class ContextLearningJobConfig:
    """Configuration for creating a context learning job."""
    
    task_app_url: str
    evaluation_seeds: List[int]
    task_app_api_key: Optional[str] = None
    environment: Optional[EnvironmentConfig] = None
    algorithm_config: Optional[AlgorithmConfig] = None
    metadata: Optional[Dict[str, Any]] = None
    org_id: Optional[str] = None
    max_concurrent_rollouts: int = 10
    verifier_base_url: Optional[str] = None
    verifier_job_id: str = "zero_shot_verifier_rubric_mapreduce"
    verifier_model: str = "gpt-4.1-mini"
    require_agent_trace_log: bool = True
    auto_start: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextLearningJobConfig:
        """Create a ContextLearningJobConfig from a dictionary."""
        env_data = data.get("environment")
        env = EnvironmentConfig.from_dict(env_data) if isinstance(env_data, dict) else None
        
        algo_data = data.get("algorithm_config")
        algo = AlgorithmConfig.from_dict(algo_data) if isinstance(algo_data, dict) else None
        
        return cls(
            task_app_url=data["task_app_url"],
            evaluation_seeds=data.get("evaluation_seeds", []),
            task_app_api_key=data.get("task_app_api_key"),
            environment=env,
            algorithm_config=algo,
            metadata=data.get("metadata"),
            org_id=data.get("org_id"),
            max_concurrent_rollouts=data.get("max_concurrent_rollouts", 10),
            verifier_base_url=data.get("verifier_base_url"),
            verifier_job_id=data.get("verifier_job_id", "zero_shot_verifier_rubric_mapreduce"),
            verifier_model=data.get("verifier_model", "gpt-4.1-mini"),
            require_agent_trace_log=data.get("require_agent_trace_log", True),
            auto_start=data.get("auto_start", True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: Dict[str, Any] = {
            "task_app_url": self.task_app_url,
            "evaluation_seeds": self.evaluation_seeds,
            "max_concurrent_rollouts": self.max_concurrent_rollouts,
            "verifier_job_id": self.verifier_job_id,
            "verifier_model": self.verifier_model,
            "require_agent_trace_log": self.require_agent_trace_log,
            "auto_start": self.auto_start,
        }
        if self.task_app_api_key:
            result["task_app_api_key"] = self.task_app_api_key
        if self.environment:
            result["environment"] = self.environment.to_dict()
        if self.algorithm_config:
            result["algorithm_config"] = self.algorithm_config.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        if self.org_id:
            result["org_id"] = self.org_id
        if self.verifier_base_url:
            result["verifier_base_url"] = self.verifier_base_url
        return result


@dataclass
class ContextLearningJobStatus:
    """Status of a context learning job."""
    
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    best_score: Optional[float] = None
    best_preflight_script: Optional[str] = None
    environment: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextLearningJobStatus:
        """Create a ContextLearningJobStatus from a dictionary."""
        return cls(
            job_id=data["job_id"],
            status=data.get("status", "unknown"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            best_score=data.get("best_score"),
            best_preflight_script=data.get("best_preflight_script"),
            environment=data.get("environment"),
            metadata=data.get("metadata"),
            recent_events=data.get("recent_events", []),
            error=data.get("error"),
        )
    
    @property
    def is_terminal(self) -> bool:
        """Check if the job is in a terminal state."""
        return self.status in {"completed", "succeeded", "failed", "cancelled"}
    
    @property
    def is_successful(self) -> bool:
        """Check if the job completed successfully."""
        return self.status in {"completed", "succeeded"}


@dataclass
class ContextLearningEvent:
    """An event from a context learning job."""
    
    event_type: str
    message: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    seq: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextLearningEvent:
        """Create a ContextLearningEvent from a dictionary."""
        return cls(
            event_type=data.get("event_type", data.get("type", "")),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", data.get("created_at", "")),
            metadata=data.get("metadata", data.get("data", {})),
            seq=data.get("seq"),
        )


@dataclass
class BestScriptResult:
    """Result containing the best performing pre-flight script."""
    
    job_id: str
    best_score: float
    preflight_script: str
    generation: int
    variation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BestScriptResult:
        """Create a BestScriptResult from a dictionary."""
        return cls(
            job_id=data.get("job_id", ""),
            best_score=float(data.get("best_score", 0.0)),
            preflight_script=str(data.get("preflight_script", "")),
            generation=int(data.get("generation", 0)),
            variation_id=str(data.get("variation_id", "")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ContextLearningMetric:
    """A metric point from context learning optimization."""
    
    name: str
    value: float
    step: int
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextLearningMetric:
        """Create a ContextLearningMetric from a dictionary."""
        return cls(
            name=data.get("name", ""),
            value=float(data.get("value", 0.0)),
            step=int(data.get("step", 0)),
            timestamp=data.get("timestamp", data.get("created_at", "")),
            data=data.get("data", {}),
        )


@dataclass
class ContextLearningResults:
    """Complete results from a context learning job."""
    
    job_id: str
    status: str
    best_score: Optional[float] = None
    best_script: Optional[BestScriptResult] = None
    generations_completed: int = 0
    events: List[ContextLearningEvent] = field(default_factory=list)
    metrics: List[ContextLearningMetric] = field(default_factory=list)
    
    @classmethod
    def from_status_and_events(
        cls,
        status: ContextLearningJobStatus,
        events: List[ContextLearningEvent],
        best_script: Optional[BestScriptResult] = None,
    ) -> ContextLearningResults:
        """Create results from status and events."""
        # Extract generation count from events
        generations_completed = 0
        for event in events:
            if event.event_type == "context.learning.generation.completed":
                gen = event.metadata.get("generation", 0)
                if isinstance(gen, int) and gen > generations_completed:
                    generations_completed = gen
        
        return cls(
            job_id=status.job_id,
            status=status.status,
            best_score=status.best_score,
            best_script=best_script,
            generations_completed=generations_completed,
            events=events,
        )


