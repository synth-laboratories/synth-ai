"""Experiment status tracking for progress reporting and monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentStatus:
    """Status object that experiments can update and pollers can display.
    
    This class represents the current status of an experiment job, including
    progress metrics, policy/environment info, and ETA estimates.
    """

    policy: str | None = None
    environment: str | None = None
    rollouts_completed: int | None = None
    total_rollouts: int | None = None
    trials_completed: int | None = None
    best_score: float | None = None
    current_trial: int | None = None
    eta_seconds: float | None = None
    progress_pct: float | None = None
    rollouts_per_minute: float | None = None
    custom_fields: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate status data after initialization.
        
        Raises:
            AssertionError: If data is invalid (e.g., rollouts_completed > total_rollouts)
        """
        # Ensure custom_fields is never None (can happen if explicitly passed as None)
        if self.custom_fields is None:
            object.__setattr__(self, 'custom_fields', {})
        
        # Validate custom_fields type
        assert isinstance(self.custom_fields, dict), (
            f"custom_fields must be dict, got {type(self.custom_fields).__name__}: {self.custom_fields}"
        )
        
        # Validate rollouts
        if self.rollouts_completed is not None:
            assert self.rollouts_completed >= 0, (
                f"rollouts_completed must be >= 0, got {self.rollouts_completed}"
            )
            if self.total_rollouts is not None:
                assert self.total_rollouts > 0, (
                    f"total_rollouts must be > 0, got {self.total_rollouts}"
                )
                assert self.rollouts_completed <= self.total_rollouts, (
                    f"rollouts_completed ({self.rollouts_completed}) > total_rollouts ({self.total_rollouts})"
                )
        
        # Validate ETA
        if self.eta_seconds is not None:
            assert self.eta_seconds >= 0, (
                f"eta_seconds must be >= 0, got {self.eta_seconds}"
            )
        
        # Validate progress percentage
        if self.progress_pct is not None:
            assert 0 <= self.progress_pct <= 100, (
                f"progress_pct must be in [0, 100], got {self.progress_pct}"
            )
        
        # Validate best_score (assume 0-1 range for accuracy)
        if self.best_score is not None:
            assert 0 <= self.best_score <= 1, (
                f"best_score must be in [0, 1], got {self.best_score}"
            )
        
        # Validate trials
        if self.trials_completed is not None:
            assert self.trials_completed >= 0, (
                f"trials_completed must be >= 0, got {self.trials_completed}"
            )
        
        if self.current_trial is not None:
            assert self.current_trial >= 0, (
                f"current_trial must be >= 0, got {self.current_trial}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON storage in database."""
        result: dict[str, Any] = {}
        if self.policy is not None:
            result["policy"] = self.policy
        if self.environment is not None:
            result["environment"] = self.environment
        if self.rollouts_completed is not None:
            result["rollouts_completed"] = self.rollouts_completed
        if self.total_rollouts is not None:
            result["total_rollouts"] = self.total_rollouts
        if self.trials_completed is not None:
            result["trials_completed"] = self.trials_completed
        if self.best_score is not None:
            result["best_score"] = self.best_score
        if self.current_trial is not None:
            result["current_trial"] = self.current_trial
        if self.eta_seconds is not None:
            result["eta_seconds"] = self.eta_seconds
        if self.progress_pct is not None:
            result["progress_pct"] = self.progress_pct
        
        # Safely update with custom_fields (ensure it's not None)
        custom_fields = self.custom_fields if self.custom_fields is not None else {}
        assert isinstance(custom_fields, dict), (
            f"custom_fields must be dict, got {type(custom_fields).__name__}: {custom_fields}"
        )
        result.update(custom_fields)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentStatus:
        """Create from dict (loaded from database) with type coercion and validation.
        
        Args:
            data: Dictionary containing status fields (may have wrong types from JSON)
            
        Returns:
            ExperimentStatus instance with validated and coerced types
            
        Raises:
            AssertionError: If data cannot be coerced or is invalid
        """
        assert isinstance(data, dict), (
            f"data must be dict, got {type(data).__name__}: {data}"
        )
        
        # Extract known fields, rest go to custom_fields
        known_fields = {
            "policy",
            "environment",
            "rollouts_completed",
            "total_rollouts",
            "trials_completed",
            "best_score",
            "current_trial",
            "eta_seconds",
            "progress_pct",
        }
        kwargs: dict[str, Any] = {}
        custom_fields: dict[str, Any] = {}
        
        # Type coercion helper
        def coerce_int(key: str, value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float | str):
                try:
                    return int(float(value))
                except (ValueError, TypeError) as e:
                    raise AssertionError(
                        f"Cannot coerce {key} to int: {value} (type: {type(value).__name__})"
                    ) from e
            raise AssertionError(
                f"{key} must be int | None, got {type(value).__name__}: {value}"
            )
        
        def coerce_float(key: str, value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError) as e:
                    raise AssertionError(
                        f"Cannot coerce {key} to float: {value} (type: {type(value).__name__})"
                    ) from e
            raise AssertionError(
                f"{key} must be float | None, got {type(value).__name__}: {value}"
            )
        
        def coerce_str(key: str, value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            # Allow coercion from other types for flexibility
            return str(value)
        
        for key, value in data.items():
            if key in known_fields:
                # Coerce types based on field
                if key in ("rollouts_completed", "total_rollouts", "trials_completed", "current_trial"):
                    kwargs[key] = coerce_int(key, value)
                elif key in ("best_score", "eta_seconds", "progress_pct"):
                    kwargs[key] = coerce_float(key, value)
                elif key in ("policy", "environment"):
                    kwargs[key] = coerce_str(key, value)
                else:
                    kwargs[key] = value
            else:
                custom_fields[key] = value
        
        # Always set custom_fields (even if empty) to ensure it's never None
        kwargs["custom_fields"] = custom_fields if custom_fields else {}
        
        # Validate custom_fields before creating instance
        assert isinstance(kwargs["custom_fields"], dict), (
            f"custom_fields must be dict, got {type(kwargs['custom_fields']).__name__}: {kwargs['custom_fields']}"
        )
        
        # Create instance (__post_init__ will validate)
        return cls(**kwargs)

    def format_eta(self) -> str | None:
        """Format ETA as human-readable string (e.g., "2m 30s", "1h 15m")."""
        if self.eta_seconds is None:
            return None
        
        seconds = int(self.eta_seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            if secs == 0:
                return f"{minutes}m"
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if minutes == 0:
                return f"{hours}h"
            return f"{hours}h {minutes}m"

    def format_progress(self) -> str | None:
        """Format progress as human-readable string (e.g., "15/200 (7.5%)")."""
        if self.rollouts_completed is None or self.total_rollouts is None:
            return None
        
        pct = ""
        if self.progress_pct is not None:
            pct = f" ({self.progress_pct:.1f}%)"
        elif self.total_rollouts > 0:
            pct_pct = (self.rollouts_completed / self.total_rollouts) * 100
            pct = f" ({pct_pct:.1f}%)"
        
        return f"{self.rollouts_completed}/{self.total_rollouts}{pct}"

    def format_status_line(self) -> str:
        """Format a status line for display in poller.
        
        Returns:
            Formatted string like: "Policy: gpt-4 | Env: heartdisease | 15/200 (7.5%) | ETA: 2m | Best: 0.85 | 12.5 rollouts/min"
        """
        parts: list[str] = []
        
        if self.policy:
            parts.append(f"Policy: {self.policy}")
        if self.environment:
            parts.append(f"Env: {self.environment}")
        
        progress = self.format_progress()
        if progress:
            parts.append(progress)
        
        eta = self.format_eta()
        if eta:
            parts.append(f"ETA: {eta}")
        
        if self.best_score is not None:
            parts.append(f"Best: {self.best_score:.3f}")
        
        if self.rollouts_per_minute is not None:
            parts.append(f"{self.rollouts_per_minute:.1f} rollouts/min")
        
        return " | ".join(parts) if parts else "No status available"


class ExperimentStatusTracker:
    """Helper for experiments to report their progress to the queue.
    
    Usage:
        tracker = ExperimentStatusTracker(job_id="job_abc123")
        tracker.update(
            policy="gpt-4",
            environment="heartdisease",
            rollouts_completed=15,
            total_rollouts=200,
            best_score=0.85,
        )
    """

    def __init__(self, job_id: str):
        """Initialize status tracker for a job.
        
        Args:
            job_id: Experiment job ID to update
        """
        self.job_id = job_id

    def update(
        self,
        policy: str | None = None,
        environment: str | None = None,
        rollouts_completed: int | None = None,
        total_rollouts: int | None = None,
        trials_completed: int | None = None,
        best_score: float | None = None,
        current_trial: int | None = None,
        eta_seconds: float | None = None,
        progress_pct: float | None = None,
        rollouts_per_minute: float | None = None,
        custom_fields: dict[str, Any] | None = None,
    ) -> None:
        """Update experiment status in the database.
        
        This method writes the status to the ExperimentJob.status_json field
        via the service layer. All data is validated before storage.
        
        Args:
            policy: Policy/model identifier
            environment: Environment/task name
            rollouts_completed: Number of rollouts completed
            total_rollouts: Total rollouts planned
            trials_completed: Number of trials completed
            best_score: Best score achieved
            current_trial: Current trial number
            eta_seconds: Estimated time to completion (seconds)
            progress_pct: Progress percentage (0-100)
            custom_fields: Additional custom fields
            
        Raises:
            AssertionError: If data is invalid (e.g., rollouts_completed > total_rollouts)
        """
        from .service import update_job_status
        
        # Validate inputs before creating ExperimentStatus
        assert isinstance(self.job_id, str), (
            f"job_id must be str, got {type(self.job_id).__name__}: {self.job_id}"
        )
        assert self.job_id, "job_id cannot be empty"
        
        # Ensure custom_fields is never None
        if custom_fields is None:
            custom_fields = {}
        assert isinstance(custom_fields, dict), (
            f"custom_fields must be dict | None, got {type(custom_fields).__name__}: {custom_fields}"
        )
        
        # Create status object (will validate in __post_init__)
        status = ExperimentStatus(
            policy=policy,
            environment=environment,
            rollouts_completed=rollouts_completed,
            total_rollouts=total_rollouts,
            trials_completed=trials_completed,
            best_score=best_score,
            current_trial=current_trial,
            eta_seconds=eta_seconds,
            progress_pct=progress_pct,
            rollouts_per_minute=rollouts_per_minute,
            custom_fields=custom_fields,
        )
        
        # Convert to dict and validate structure
        status_dict = status.to_dict()
        assert isinstance(status_dict, dict), (
            f"status.to_dict() must return dict, got {type(status_dict).__name__}"
        )
        
        # Update database
        update_job_status(self.job_id, status_dict)
