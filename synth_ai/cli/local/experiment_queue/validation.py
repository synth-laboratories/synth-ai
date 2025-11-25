"""Validation utilities and decorators for type safety and data integrity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def validate_not_none(value: Any, name: str) -> None:
    """Assert that a value is not None.
    
    Args:
        value: Value to check
        name: Name of the value (for error messages)
        
    Raises:
        AssertionError: If value is None
    """
    assert value is not None, f"{name} cannot be None"


def validate_non_empty_string(value: str | None, name: str) -> str:
    """Validate that a string is not None and not empty.
    
    Args:
        value: String to validate
        name: Name of the value (for error messages)
        
    Returns:
        The validated string
        
    Raises:
        AssertionError: If value is None or empty
    """
    assert value is not None, f"{name} cannot be None"
    assert isinstance(value, str), f"{name} must be str, got {type(value).__name__}"
    assert value.strip(), f"{name} cannot be empty"
    return value


def validate_positive_int(value: int | None, name: str, allow_none: bool = True) -> int | None:
    """Validate that an integer is positive (if not None).
    
    Args:
        value: Integer to validate
        name: Name of the value (for error messages)
        allow_none: Whether None is allowed
        
    Returns:
        The validated integer or None
        
    Raises:
        AssertionError: If value is invalid
    """
    if value is None:
        assert allow_none, f"{name} cannot be None"
        return None
    assert isinstance(value, int), f"{name} must be int, got {type(value).__name__}: {value}"
    assert value >= 0, f"{name} must be >= 0, got {value}"
    return value


def validate_range(
    value: int | float | None,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    allow_none: bool = True,
) -> int | float | None:
    """Validate that a numeric value is within a range.
    
    Args:
        value: Numeric value to validate
        name: Name of the value (for error messages)
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_none: Whether None is allowed
        
    Returns:
        The validated value or None
        
    Raises:
        AssertionError: If value is out of range
    """
    if value is None:
        assert allow_none, f"{name} cannot be None"
        return None
    assert isinstance(value, int | float), (
        f"{name} must be int | float, got {type(value).__name__}: {value}"
    )
    if min_val is not None:
        assert value >= min_val, f"{name} must be >= {min_val}, got {value}"
    if max_val is not None:
        assert value <= max_val, f"{name} must be <= {max_val}, got {value}"
    return value


def validate_dict(value: dict[str, Any] | None, name: str, allow_none: bool = True) -> dict[str, Any]:
    """Validate that a value is a dict (or None if allowed).
    
    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        allow_none: Whether None is allowed (returns empty dict if None)
        
    Returns:
        The validated dict (or empty dict if None and allow_none=True)
        
    Raises:
        AssertionError: If value is not a dict
    """
    if value is None:
        assert allow_none, f"{name} cannot be None"
        return {}
    assert isinstance(value, dict), (
        f"{name} must be dict, got {type(value).__name__}: {value}"
    )
    return value


def validate_path(value: str | Path, name: str, must_exist: bool = False) -> Path:
    """Validate that a path is valid and optionally exists.
    
    Args:
        value: Path to validate
        name: Name of the value (for error messages)
        must_exist: Whether the path must exist
        
    Returns:
        The validated Path object
        
    Raises:
        AssertionError: If path is invalid or doesn't exist (if required)
    """
    from pathlib import Path
    
    assert value is not None, f"{name} cannot be None"
    path = Path(value) if isinstance(value, str) else value
    assert isinstance(path, Path), f"{name} must be str or Path, got {type(value).__name__}"
    if must_exist:
        assert path.exists(), f"{name} path does not exist: {path}"
    return path


def validated(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add validation to function arguments.
    
    This is a placeholder for future validation decorator implementation.
    Currently just passes through the function unchanged.
    """
    return func




