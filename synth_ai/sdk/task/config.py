"""Configuration dataclasses for task app CLI commands (eval, filter)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class EvalConfig:
    """Configuration for 'synth-ai eval' command.
    
    Validates and provides defaults for evaluation runs against task apps.
    """
    
    # Required: Task app identifier
    app_id: str
    
    # Required: Model to evaluate
    model: str
    
    # Required: Seeds to run
    seeds: list[int]
    
    # Optional: Task app URL (None = spawn in-process)
    task_app_url: str | None = None
    
    # Optional: Data split to use
    split: str = "train"
    
    # Optional: Maximum turns/steps per episode
    max_turns: int | None = None
    
    # Optional: Maximum LLM calls per episode
    max_llm_calls: int = 10
    
    # Optional: Concurrency for parallel rollouts
    concurrency: int = 1
    
    # Optional: Environment name
    env_name: str | None = None
    
    # Optional: Policy name
    policy_name: str | None = None
    
    # Optional: Trace format ("compact", "full", "structured")
    trace_format: Literal["compact", "full", "structured"] = "compact"
    
    # Optional: Whether to return traces in response
    return_trace: bool = False
    
    # Optional: Operations sequence (if not provided, generates default)
    ops: list[str] | None = None
    
    # Optional: Environment config overrides
    env_config: dict[str, Any] = field(default_factory=dict)
    
    # Optional: Policy config overrides
    policy_config: dict[str, Any] = field(default_factory=dict)
    
    # Optional: Metadata for traces
    metadata: dict[str, str] = field(default_factory=dict)
    
    # Optional: SQL query for metadata filtering
    metadata_sql: str | None = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.app_id:
            raise ValueError("app_id is required")
        
        if not self.model:
            raise ValueError("model is required")
        
        if not self.seeds:
            raise ValueError("seeds list cannot be empty")
        
        if not isinstance(self.seeds, list):
            raise ValueError("seeds must be a list of integers")
        
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        
        if self.max_llm_calls < 1:
            raise ValueError("max_llm_calls must be >= 1")
        
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        
        if self.trace_format not in ("compact", "full", "structured"):
            raise ValueError(f"trace_format must be 'compact', 'full', or 'structured', got: {self.trace_format}")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        """Create EvalConfig from a dictionary (e.g. from TOML).
        
        Args:
            data: Dictionary with eval configuration
            
        Returns:
            Validated EvalConfig instance
        """
        # Extract known fields
        config_dict = {
            "app_id": data.get("app_id"),
            "model": data.get("model"),
            "seeds": data.get("seeds", []),
            "task_app_url": data.get("task_app_url"),
            "split": data.get("split", "train"),
            "max_turns": data.get("max_turns"),
            "max_llm_calls": data.get("max_llm_calls", 10),
            "concurrency": data.get("concurrency", 1),
            "env_name": data.get("env_name"),
            "policy_name": data.get("policy_name"),
            "trace_format": data.get("trace_format", "compact"),
            "return_trace": data.get("return_trace", False),
            "ops": data.get("ops"),
            "env_config": data.get("env_config", {}),
            "policy_config": data.get("policy_config", {}),
            "metadata": data.get("metadata", {}),
            "metadata_sql": data.get("metadata_sql"),
        }
        
        return cls(**config_dict)


@dataclass(slots=True)
class FilterConfig:
    """Configuration for 'synth-ai filter' command.
    
    Validates and provides defaults for filtering traces into SFT datasets.
    """
    
    # Required: Database path or URL
    db: str
    
    # Required: Output JSONL path
    output: str
    
    # Optional: Filter by data splits
    splits: list[str] = field(default_factory=list)
    
    # Optional: Filter by task IDs
    task_ids: list[str] = field(default_factory=list)
    
    # Optional: Filter by models
    models: list[str] = field(default_factory=list)
    
    # Optional: Minimum official score threshold
    min_official_score: float | None = None
    
    # Optional: Maximum official score threshold
    max_official_score: float | None = None
    
    # Optional: Minimum judge scores (judge_name -> min_score)
    min_judge_scores: dict[str, float] = field(default_factory=dict)
    
    # Optional: Maximum judge scores (judge_name -> max_score)
    max_judge_scores: dict[str, float] = field(default_factory=dict)
    
    # Optional: Limit number of examples
    limit: int | None = None
    
    # Optional: Offset for pagination
    offset: int | None = None
    
    # Optional: Whether to shuffle results
    shuffle: bool = False
    
    # Optional: Random seed for shuffling
    shuffle_seed: int | None = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.db:
            raise ValueError("db (database path or URL) is required")
        
        if not self.output:
            raise ValueError("output (JSONL file path) is required")
        
        # Validate output has .jsonl extension
        output_path = Path(self.output)
        if output_path.suffix.lower() not in (".jsonl", ".json"):
            raise ValueError(f"output must be a .jsonl or .json file, got: {self.output}")
        
        # Validate score thresholds
        if (
            self.min_official_score is not None
            and self.max_official_score is not None
            and self.min_official_score > self.max_official_score
        ):
            raise ValueError("min_official_score cannot be greater than max_official_score")
        
        # Validate limit/offset
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1")
        
        if self.offset is not None and self.offset < 0:
            raise ValueError("offset must be >= 0")
        
        # Validate shuffle seed requires shuffle
        if self.shuffle_seed is not None and not self.shuffle:
            raise ValueError("shuffle_seed requires shuffle=true")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterConfig:
        """Create FilterConfig from a dictionary (e.g. from TOML).
        
        Args:
            data: Dictionary with filter configuration
            
        Returns:
            Validated FilterConfig instance
        """
        # Extract known fields
        config_dict = {
            "db": data.get("db"),
            "output": data.get("output"),
            "splits": data.get("splits", []),
            "task_ids": data.get("task_ids", []),
            "models": data.get("models", []),
            "min_official_score": data.get("min_official_score"),
            "max_official_score": data.get("max_official_score"),
            "min_judge_scores": data.get("min_judge_scores", {}),
            "max_judge_scores": data.get("max_judge_scores", {}),
            "limit": data.get("limit"),
            "offset": data.get("offset"),
            "shuffle": data.get("shuffle", False),
            "shuffle_seed": data.get("shuffle_seed"),
        }
        
        return cls(**config_dict)
    
    def get_db_url(self) -> str:
        """Convert db path to proper SQLite URL if needed.
        
        Returns:
            Database URL suitable for SQLAlchemy/aiosqlite
        """
        db_value = self.db.strip()
        if "://" in db_value:
            return db_value
        else:
            db_path = Path(db_value).expanduser().resolve()
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
    
    def get_output_path(self) -> Path:
        """Get resolved output path with parent directory created.
        
        Returns:
            Resolved Path object with parent directory created
        """
        output_path = Path(self.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path



