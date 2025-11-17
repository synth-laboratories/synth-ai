"""Type-safe progress information dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgressInfo:
    """Type-safe representation of progress information parsed from output.
    
    This replaces dict[str, Any] usage in status_tracker.py for better type safety.
    """

    rollouts_completed: int | None = None
    total_rollouts: int | None = None
    best_score: float | None = None
    trials_completed: int | None = None

    def __post_init__(self) -> None:
        """Validate progress data after initialization."""
        # Validate rollouts_completed
        if self.rollouts_completed is not None:
            assert isinstance(self.rollouts_completed, int), (
                f"rollouts_completed must be int, got {type(self.rollouts_completed).__name__}"
            )
            assert self.rollouts_completed >= 0, (
                f"rollouts_completed must be >= 0, got {self.rollouts_completed}"
            )
            if self.total_rollouts is not None:
                assert self.rollouts_completed <= self.total_rollouts, (
                    f"rollouts_completed ({self.rollouts_completed}) > total_rollouts ({self.total_rollouts})"
                )

        # Validate total_rollouts
        if self.total_rollouts is not None:
            assert isinstance(self.total_rollouts, int), (
                f"total_rollouts must be int, got {type(self.total_rollouts).__name__}"
            )
            assert self.total_rollouts > 0, (
                f"total_rollouts must be > 0, got {self.total_rollouts}"
            )

        # Validate best_score
        if self.best_score is not None:
            assert isinstance(self.best_score, int | float), (
                f"best_score must be int | float, got {type(self.best_score).__name__}"
            )
            assert 0 <= self.best_score <= 1, (
                f"best_score must be in [0, 1], got {self.best_score}"
            )

        # Validate trials_completed
        if self.trials_completed is not None:
            assert isinstance(self.trials_completed, int), (
                f"trials_completed must be int, got {type(self.trials_completed).__name__}"
            )
            assert self.trials_completed >= 0, (
                f"trials_completed must be >= 0, got {self.trials_completed}"
            )

    def to_dict(self) -> dict[str, int | float | None]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "rollouts_completed": self.rollouts_completed,
            "total_rollouts": self.total_rollouts,
            "best_score": self.best_score,
            "trials_completed": self.trials_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressInfo:
        """Create from dictionary with type coercion and validation.
        
        Args:
            data: Dictionary with progress information (may have wrong types)
            
        Returns:
            ProgressInfo instance with validated and coerced types
            
        Raises:
            AssertionError: If data cannot be coerced or is invalid
        """
        # Coerce and validate each field
        rollouts_completed = None
        if "rollouts_completed" in data and data["rollouts_completed"] is not None:
            val = data["rollouts_completed"]
            if isinstance(val, int | float | str):
                try:
                    coerced = int(float(val))
                    assert coerced >= 0, f"rollouts_completed must be >= 0, got {coerced}"
                    rollouts_completed = coerced
                except (ValueError, TypeError, AssertionError) as e:
                    raise AssertionError(
                        f"Cannot coerce rollouts_completed to int: {val} (type: {type(val).__name__})"
                    ) from e
            else:
                raise AssertionError(
                    f"rollouts_completed must be int | float | str, got {type(val).__name__}: {val}"
                )

        total_rollouts = None
        if "total_rollouts" in data and data["total_rollouts"] is not None:
            val = data["total_rollouts"]
            if isinstance(val, int | float | str):
                try:
                    total_rollouts = int(float(val))
                except (ValueError, TypeError) as e:
                    raise AssertionError(
                        f"Cannot coerce total_rollouts to int: {val} (type: {type(val).__name__})"
                    ) from e
            else:
                raise AssertionError(
                    f"total_rollouts must be int | float | str, got {type(val).__name__}: {val}"
                )
            if total_rollouts is not None:
                assert total_rollouts > 0, f"total_rollouts must be > 0, got {total_rollouts}"

        best_score = None
        if "best_score" in data and data["best_score"] is not None:
            val = data["best_score"]
            if isinstance(val, int | float | str):
                try:
                    coerced = float(val)
                    assert 0 <= coerced <= 1, f"best_score must be in [0, 1], got {coerced}"
                    best_score = coerced
                except (ValueError, TypeError, AssertionError) as e:
                    raise AssertionError(
                        f"Cannot coerce best_score to float: {val} (type: {type(val).__name__})"
                    ) from e
            else:
                raise AssertionError(
                    f"best_score must be int | float | str, got {type(val).__name__}: {val}"
                )

        trials_completed = None
        if "trials_completed" in data and data["trials_completed"] is not None:
            val = data["trials_completed"]
            if isinstance(val, int | float | str):
                try:
                    coerced = int(float(val))
                    assert coerced >= 0, f"trials_completed must be >= 0, got {coerced}"
                    trials_completed = coerced
                except (ValueError, TypeError, AssertionError) as e:
                    raise AssertionError(
                        f"Cannot coerce trials_completed to int: {val} (type: {type(val).__name__})"
                    ) from e
            else:
                raise AssertionError(
                    f"trials_completed must be int | float | str, got {type(val).__name__}: {val}"
                )

        return cls(
            rollouts_completed=rollouts_completed,
            total_rollouts=total_rollouts,
            best_score=best_score,
            trials_completed=trials_completed,
        )

