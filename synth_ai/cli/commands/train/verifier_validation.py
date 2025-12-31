"""
Validation logic for verifier/rubric configuration from TOML.

This module validates and normalizes verifier/rubric config, removing all dead fields
and ensuring only the fields actually used by the backend are present.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Optional, Tuple

from pydantic import ValidationError

from .errors import InvalidRubricConfigError, InvalidVerifierConfigError
from .verifier_schemas import (
    RubricConfig,
    RubricWeightsConfig,
    VerifierConfig,
    VerifierOptionsConfig,
)

__all__ = [
    "validate_verifier_config",
    "validate_rubric_config",
    "extract_and_validate_verifier_rubric",
]

# Dead fields that should trigger deprecation warnings
DEPRECATED_RUBRIC_FIELDS = {
    "model",
    "api_base",
    "api_key_env",
    "event",
    "outcome",
}

DEPRECATED_VERIFIER_FIELDS = {
    "type",
    "timeout_s",  # Moved to verifier.options.timeout_s
}

DEPRECATED_VERIFIER_OPTIONS_FIELDS = {
    "max_concurrency",
    "tracks",
}


def _reject_deprecated_fields(
    section: str,
    fields: set[str],
    present_fields: set[str],
    error_cls: type[Exception],
) -> None:
    deprecated_present = fields & present_fields
    if deprecated_present:
        field_list = ", ".join(sorted(deprecated_present))
        raise error_cls(
            detail=f"[{section}] contains deprecated fields that are not supported: {field_list}."
        )


def validate_rubric_config(config: MutableMapping[str, Any]) -> RubricConfig:
    """
    Validate and normalize rubric configuration from TOML.
    
    Args:
        config: Raw [rubric] section from TOML
        
    Returns:
        Validated RubricConfig instance
        
    Raises:
        InvalidRubricConfigError: If validation fails
    """
    if not config:
        # Default: rubric disabled
        return RubricConfig(enabled=False)
    
    config_dict = dict(config)
    
    _reject_deprecated_fields(
        "rubric",
        DEPRECATED_RUBRIC_FIELDS,
        set(config_dict.keys()),
        InvalidRubricConfigError,
    )
    
    if "event" in config_dict:
        raise InvalidRubricConfigError(
            detail="[rubric.event] is not supported. Use [verifier.options.rubric_overrides] instead."
        )
    
    if "outcome" in config_dict:
        raise InvalidRubricConfigError(
            detail="[rubric.outcome] is not supported. Use [verifier.options.rubric_overrides] instead."
        )
    
    # Extract only valid fields
    enabled = config_dict.get("enabled", False)
    weights_dict = config_dict.get("weights", {})
    
    # Validate using Pydantic
    try:
        if not isinstance(weights_dict, dict):
            raise ValueError("[rubric.weights] must be a dictionary")
        
        weights = RubricWeightsConfig(**weights_dict)
        return RubricConfig(enabled=enabled, weights=weights)
    
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  • rubric.{loc}: {msg}")
        raise InvalidRubricConfigError(
            detail="Rubric validation failed:\n" + "\n".join(errors)
        ) from exc
    except Exception as exc:
        raise InvalidRubricConfigError(
            detail=f"Rubric validation failed: {exc}"
        ) from exc


def validate_verifier_config(config: MutableMapping[str, Any]) -> Optional[VerifierConfig]:
    """
    Validate and normalize verifier configuration from TOML.
    
    Args:
        config: Raw [verifier] section from TOML
        
    Returns:
        Validated VerifierConfig instance, or None if not present
        
    Raises:
        InvalidVerifierConfigError: If validation fails
    """
    if not config:
        return None
    
    config_dict = dict(config)
    
    _reject_deprecated_fields(
        "verifier",
        DEPRECATED_VERIFIER_FIELDS,
        set(config_dict.keys()),
        InvalidVerifierConfigError,
    )
    
    # Extract verifier.options (required)
    options_dict = config_dict.get("options")
    if not options_dict:
        raise InvalidVerifierConfigError(
            detail="[verifier.options] section is required when [verifier] is present"
        )
    
    if not isinstance(options_dict, dict):
        raise InvalidVerifierConfigError(
            detail="[verifier.options] must be a dictionary"
        )
    
    _reject_deprecated_fields(
        "verifier.options",
        DEPRECATED_VERIFIER_OPTIONS_FIELDS,
        set(options_dict.keys()),
        InvalidVerifierConfigError,
    )
    
    # Validate using Pydantic
    try:
        options = VerifierOptionsConfig(**options_dict)
        return VerifierConfig(options=options)
    
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  • verifier.options.{loc}: {msg}")
        raise InvalidVerifierConfigError(
            detail="Verifier validation failed:\n" + "\n".join(errors)
        ) from exc
    except Exception as exc:
        raise InvalidVerifierConfigError(
            detail=f"Verifier validation failed: {exc}"
        ) from exc


def extract_and_validate_verifier_rubric(
    toml_config: MutableMapping[str, Any]
) -> Tuple[RubricConfig, Optional[VerifierConfig]]:
    """
    Extract and validate verifier/rubric config from full TOML config.
    
    Args:
        toml_config: Full TOML configuration dict
        
    Returns:
        Tuple of (validated_rubric, validated_verifier_or_none)
        
    Raises:
        InvalidRubricConfigError: If rubric validation fails
        InvalidVerifierConfigError: If verifier validation fails
    """
    rubric_dict = toml_config.get("rubric", {})
    verifier_dict = toml_config.get("verifier", {})
    
    # Validate rubric
    rubric_config = validate_rubric_config(rubric_dict)
    
    # Validate verifier (if present)
    verifier_config = validate_verifier_config(verifier_dict) if verifier_dict else None
    
    if rubric_config.enabled and not verifier_config:
        raise InvalidVerifierConfigError(
            detail="[rubric].enabled=true requires a [verifier] section."
        )
    
    if rubric_config.enabled and verifier_config:
        weights = rubric_config.weights
        options = verifier_config.options
        
        if weights.event > 0 and not options.event:
            raise InvalidVerifierConfigError(
                detail="[rubric.weights].event > 0 requires [verifier.options].event=true."
            )
        
        if weights.outcome > 0 and not options.outcome:
            raise InvalidVerifierConfigError(
                detail="[rubric.weights].outcome > 0 requires [verifier.options].outcome=true."
            )
    
    return rubric_config, verifier_config

