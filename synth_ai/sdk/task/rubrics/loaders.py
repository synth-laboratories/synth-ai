"""Rubric loading and blending utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Criterion, Rubric


def _load_text(source: str) -> tuple[str, str | None]:
    """Load text from file path or return as-is."""
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8"), path.suffix.lower()
    return source, None


def _parse_structured(text: str, suffix: str | None) -> dict[str, Any]:
    """Parse JSON or YAML text into a dictionary."""
    text = text.strip()
    if not text:
        raise ValueError("Rubric source is empty")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to load YAML rubrics") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Rubric YAML must produce a mapping") from None
        return data
    if text.startswith("{"):
        return json.loads(text)
    if text.startswith("http://") or text.startswith("https://"):
        import requests  # type: ignore

        response = requests.get(text, timeout=15)
        response.raise_for_status()
        return _parse_structured(response.text, suffix)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to load rubric text") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Rubric text must decode to a mapping") from None
        return data


def load_rubric(source: str | dict[str, Any] | Rubric | None) -> Rubric | None:
    """Load rubric from file path, dict, or return existing Rubric.
    
    Args:
        source: File path (JSON/YAML), dict, existing Rubric, or None
        
    Returns:
        Parsed Rubric instance or None if source is None
        
    Raises:
        ValueError: If the rubric format is incorrect (e.g., backend judge format)
        ValidationError: If the rubric fails schema validation
    """
    if source is None:
        return None
    if isinstance(source, Rubric):
        return source
    
    # Load and parse the data
    if isinstance(source, dict):
        data = source
    else:
        text, suffix = _load_text(str(source))
        data = _parse_structured(text, suffix)
    
    # Check if this looks like a backend judge rubric (wrong format)
    if (
        isinstance(data, dict)
        and "event" in data
        and "outcome" in data
        and "version" not in data
        and "goal_text" not in data
        and "criteria" not in data
    ):
        source_hint = f" ({source})" if isinstance(source, str) else ""
        raise ValueError(
            f"Rubric appears to be in backend judge format (has 'event'/'outcome' keys){source_hint}. "
            f"Task apps require rubrics with 'version', 'goal_text', and 'criteria' fields. "
            f"Backend judge rubrics should be named '*_backend_judge.json' and loaded by judge functions."
        )
    
    return Rubric.model_validate(data)


def _merge_weights(base: Criterion, override: Criterion) -> float:
    """Merge criterion weights from base and override rubrics."""
    if override.weight != 1.0 and base.weight != 1.0:
        return base.weight * override.weight
    if override.weight != 1.0:
        return override.weight
    return base.weight


def blend_rubrics(base: Rubric | None, override: Rubric | None) -> Rubric | None:
    """Blend two rubrics by merging criteria and inheriting properties.
    
    Override rubric takes precedence for descriptions and settings.
    Weights are merged multiplicatively when both are non-default.
    
    Args:
        base: Base rubric providing defaults
        override: Override rubric with specific customizations
        
    Returns:
        Blended rubric or None if both inputs are None
    """
    if override is None and base is None:
        return None
    if base is None:
        return override
    if override is None:
        return base

    base_map = {criterion.id: criterion for criterion in base.criteria}
    merged: list[Criterion] = []

    for ov in override.criteria:
        if ov.id in base_map:
            existing = base_map.pop(ov.id)
            merged.append(
                Criterion(
                    id=ov.id,
                    description=ov.description or existing.description,
                    weight=_merge_weights(existing, ov),
                    required=ov.required if ov.required is not None else existing.required,
                )
            )
        else:
            merged.append(ov)

    merged.extend(base_map.values())

    aggregation = override.aggregation
    if aggregation == "inherit":
        aggregation = base.aggregation

    return Rubric(
        version=override.version or base.version,
        goal_text=override.goal_text or base.goal_text,
        criteria=merged,
        aggregation=aggregation,
    )
