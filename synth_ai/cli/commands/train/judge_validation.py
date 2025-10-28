"""
Validation logic for judge/rubric configuration from TOML.

This module validates and normalizes judge/rubric config, removing all dead fields
and ensuring only the fields actually used by the backend are present.
"""

from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from typing import Any, Optional, Tuple

from pydantic import ValidationError

from .errors import InvalidJudgeConfigError, InvalidRubricConfigError
from .judge_schemas import JudgeConfig, JudgeOptionsConfig, RubricConfig, RubricWeightsConfig

__all__ = [
    "validate_judge_config",
    "validate_rubric_config",
    "extract_and_validate_judge_rubric",
]

# Dead fields that should trigger deprecation warnings
DEPRECATED_RUBRIC_FIELDS = {
    "model",
    "api_base",
    "api_key_env",
    "event",
    "outcome",
}

DEPRECATED_JUDGE_FIELDS = {
    "type",
    "timeout_s",  # Moved to judge.options.timeout_s
}

DEPRECATED_JUDGE_OPTIONS_FIELDS = {
    "max_concurrency",
    "tracks",
}


def _warn_deprecated_fields(section: str, fields: set[str], present_fields: set[str]) -> None:
    """Emit deprecation warnings for dead fields that are present in config."""
    deprecated_present = fields & present_fields
    if deprecated_present:
        field_list = ", ".join(sorted(deprecated_present))
        warnings.warn(
            f"[{section}] contains deprecated fields that are no longer used: {field_list}. "
            f"These fields will be ignored and should be removed from your config. "
            f"See judge/rubric cleanup guide for details.",
            DeprecationWarning,
            stacklevel=3,
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
    
    # Warn about deprecated fields
    _warn_deprecated_fields("rubric", DEPRECATED_RUBRIC_FIELDS, set(config_dict.keys()))
    
    # Warn about deprecated subsections
    if "event" in config_dict:
        warnings.warn(
            "[rubric.event] section is deprecated and no longer used. "
            "Criteria are now fetched dynamically from TaskInfo or specified in "
            "[judge.options.rubric_overrides]. This section will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    
    if "outcome" in config_dict:
        warnings.warn(
            "[rubric.outcome] section is deprecated and no longer used. "
            "Criteria are now fetched dynamically from TaskInfo or specified in "
            "[judge.options.rubric_overrides]. This section will be ignored.",
            DeprecationWarning,
            stacklevel=2,
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


def validate_judge_config(config: MutableMapping[str, Any]) -> Optional[JudgeConfig]:
    """
    Validate and normalize judge configuration from TOML.
    
    Args:
        config: Raw [judge] section from TOML
        
    Returns:
        Validated JudgeConfig instance, or None if not present
        
    Raises:
        InvalidJudgeConfigError: If validation fails
    """
    if not config:
        return None
    
    config_dict = dict(config)
    
    # Warn about deprecated top-level fields
    _warn_deprecated_fields("judge", DEPRECATED_JUDGE_FIELDS, set(config_dict.keys()))
    
    # Extract judge.options (required)
    options_dict = config_dict.get("options")
    if not options_dict:
        raise InvalidJudgeConfigError(
            detail="[judge.options] section is required when [judge] is present"
        )
    
    if not isinstance(options_dict, dict):
        raise InvalidJudgeConfigError(
            detail="[judge.options] must be a dictionary"
        )
    
    # Warn about deprecated options fields
    _warn_deprecated_fields(
        "judge.options",
        DEPRECATED_JUDGE_OPTIONS_FIELDS,
        set(options_dict.keys()),
    )
    
    # Remove deprecated fields from options
    options_dict = {
        k: v for k, v in options_dict.items()
        if k not in DEPRECATED_JUDGE_OPTIONS_FIELDS
    }
    
    # Migrate judge.timeout_s to judge.options.timeout_s if present
    if "timeout_s" in config_dict and "timeout_s" not in options_dict:
        warnings.warn(
            "[judge].timeout_s is deprecated. Use [judge.options].timeout_s instead. "
            "Auto-migrating for now.",
            DeprecationWarning,
            stacklevel=2,
        )
        options_dict["timeout_s"] = config_dict["timeout_s"]
    
    # Validate using Pydantic
    try:
        options = JudgeOptionsConfig(**options_dict)
        return JudgeConfig(options=options)
    
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  • judge.options.{loc}: {msg}")
        raise InvalidJudgeConfigError(
            detail="Judge validation failed:\n" + "\n".join(errors)
        ) from exc
    except Exception as exc:
        raise InvalidJudgeConfigError(
            detail=f"Judge validation failed: {exc}"
        ) from exc


def extract_and_validate_judge_rubric(
    toml_config: MutableMapping[str, Any]
) -> Tuple[RubricConfig, Optional[JudgeConfig]]:
    """
    Extract and validate judge/rubric config from full TOML config.
    
    Args:
        toml_config: Full TOML configuration dict
        
    Returns:
        Tuple of (validated_rubric, validated_judge_or_none)
        
    Raises:
        InvalidRubricConfigError: If rubric validation fails
        InvalidJudgeConfigError: If judge validation fails
    """
    rubric_dict = toml_config.get("rubric", {})
    judge_dict = toml_config.get("judge", {})
    
    # Validate rubric
    rubric_config = validate_rubric_config(rubric_dict)
    
    # Validate judge (if present)
    judge_config = validate_judge_config(judge_dict) if judge_dict else None
    
    # Cross-validation: If rubric is enabled, judge options should be present
    if rubric_config.enabled and not judge_config:
        warnings.warn(
            "[rubric].enabled=true but [judge] section is missing. "
            "Rubric-based judging requires judge configuration. "
            "Rubric scoring will be disabled.",
            UserWarning,
            stacklevel=2,
        )
        rubric_config.enabled = False
    
    # Cross-validation: Warn if weights don't align with enabled judging types
    if rubric_config.enabled and judge_config:
        weights = rubric_config.weights
        options = judge_config.options
        
        if weights.event > 0 and not options.event:
            warnings.warn(
                "[rubric.weights].event > 0 but [judge.options].event=false. "
                "Event-level judge scores will be 0 (no event judging enabled).",
                UserWarning,
                stacklevel=2,
            )
        
        if weights.outcome > 0 and not options.outcome:
            warnings.warn(
                "[rubric.weights].outcome > 0 but [judge.options].outcome=false. "
                "Outcome judge score will be 0 (no outcome judging enabled).",
                UserWarning,
                stacklevel=2,
            )
    
    return rubric_config, judge_config


# Helper to check if config has any deprecated fields (for testing/migration)

def check_for_deprecated_fields(toml_config: MutableMapping[str, Any]) -> dict[str, list[str]]:
    """
    Check TOML config for deprecated fields without validation.
    
    Returns dict of {section: [deprecated_field_names]} for reporting.
    """
    deprecated: dict[str, list[str]] = {}
    
    rubric_dict = toml_config.get("rubric", {})
    if rubric_dict:
        found = [
            field for field in DEPRECATED_RUBRIC_FIELDS
            if field in rubric_dict
        ]
        if "event" in rubric_dict:
            found.append("event (entire section)")
        if "outcome" in rubric_dict:
            found.append("outcome (entire section)")
        if found:
            deprecated["rubric"] = found
    
    judge_dict = toml_config.get("judge", {})
    if judge_dict:
        found = [
            field for field in DEPRECATED_JUDGE_FIELDS
            if field in judge_dict
        ]
        if found:
            deprecated["judge"] = found
        
        options_dict = judge_dict.get("options", {})
        if options_dict:
            options_found = [
                field for field in DEPRECATED_JUDGE_OPTIONS_FIELDS
                if field in options_dict
            ]
            if options_found:
                deprecated["judge.options"] = options_found
    
    return deprecated

