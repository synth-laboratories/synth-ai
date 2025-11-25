"""Base configuration classes for Synth AI SDK.

This module defines the base config class that all job configs inherit from,
ensuring consistent handling of common fields like API keys and backend URLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synth_ai.core.env import get_api_key, get_backend_url
from synth_ai.core.errors import ConfigError


@dataclass
class BaseJobConfig:
    """Base class for all job configuration dataclasses.
    
    This class provides common functionality shared across all job types
    (prompt learning, SFT, RL, research agent). Subclasses should inherit
    from this and add job-specific fields.
    
    The base class ensures consistent handling of:
    - Backend URL resolution
    - API key validation
    - Common configuration patterns
    
    Example:
        >>> from synth_ai.core.config import BaseJobConfig
        >>> from dataclasses import dataclass
        >>> 
        >>> @dataclass
        ... class MyJobConfig(BaseJobConfig):
        ...     model: str
        ...     temperature: float = 0.7
    """
    """Base configuration shared by all job types.

    All job configs should inherit from this class to ensure consistent
    handling of common fields like API keys and backend URLs.

    Attributes:
        task_app_url: URL of the task app to use
        backend_url: Synth backend URL (defaults to production)
        api_key: Synth API key (resolved from env if not provided)
        environment_api_key: API key for environment access
        timeout_seconds: Job timeout in seconds
        metadata: Optional metadata dict attached to job
    """

    task_app_url: str | None = None
    backend_url: str | None = None
    api_key: str | None = None
    environment_api_key: str | None = None
    timeout_seconds: int = 3600
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_credentials(self) -> tuple[str, str]:
        """Resolve API key and backend URL.

        Uses provided values or falls back to environment resolution.

        Returns:
            Tuple of (backend_url, api_key)

        Raises:
            ConfigError: If required credentials cannot be resolved
        """
        api_key = self.api_key or get_api_key(required=True)
        backend_url = self.backend_url or get_backend_url()

        if not api_key:
            raise ConfigError("API key is required")

        return backend_url, api_key

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        return errors


class ConfigValidator:
    """Base class for configuration validators.
    
    Validators provide a way to validate and normalize configuration
    before it's used to create jobs. This is useful for:
    - Type checking
    - Value validation
    - Default value injection
    - Cross-field validation
    
    Subclasses should implement the `validate()` method to perform
    job-specific validation logic.
    
    Example:
        >>> from synth_ai.core.config import ConfigValidator
        >>> 
        >>> class MyJobValidator(ConfigValidator):
        ...     def validate(self, config: dict) -> dict:
        ...         if config.get("temperature", 0) < 0:
        ...             raise ValueError("temperature must be >= 0")
        ...         return config
    """
    """Utility for validating configurations."""

    @staticmethod
    def require_field(
        config: dict[str, Any],
        field_name: str,
        field_type: type | tuple[type, ...] | None = None,
    ) -> Any:
        """Require a field to be present in config.

        Args:
            config: Configuration dict
            field_name: Name of required field
            field_type: Optional type(s) to validate against

        Returns:
            The field value

        Raises:
            ConfigError: If field is missing or wrong type
        """
        if field_name not in config:
            raise ConfigError(f"Missing required field: {field_name}")

        value = config[field_name]
        if field_type is not None and not isinstance(value, field_type):
            expected = (
                field_type.__name__
                if isinstance(field_type, type)
                else " | ".join(t.__name__ for t in field_type)
            )
            raise ConfigError(
                f"Field '{field_name}' must be {expected}, got {type(value).__name__}"
            )

        return value

    @staticmethod
    def validate_positive(value: int | float, field_name: str) -> None:
        """Validate that a numeric value is positive.

        Args:
            value: Value to check
            field_name: Name for error message

        Raises:
            ConfigError: If value is not positive
        """
        if value <= 0:
            raise ConfigError(f"{field_name} must be positive, got {value}")


__all__ = [
    "BaseJobConfig",
    "ConfigValidator",
]

