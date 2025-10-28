"""Rubric schema, loading, and scoring helpers for Task Apps."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Criterion(BaseModel):
    id: str
    description: str
    weight: float = 1.0
    required: bool = False

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("criterion weight must be positive")
        return value


class Rubric(BaseModel):
    version: str
    goal_text: str | None = None
    criteria: list[Criterion] = Field(default_factory=list)
    aggregation: str = "weighted_sum"

    @field_validator("aggregation")
    @classmethod
    def _validate_aggregation(cls, value: str) -> str:
        allowed = {"sum", "weighted_sum", "custom", "inherit"}
        if value not in allowed:
            raise ValueError(f"aggregation must be one of {sorted(allowed)}")
        return value

    @field_validator("criteria")
    @classmethod
    def _validate_criteria(cls, criteria: list[Criterion]) -> list[Criterion]:
        seen = set()
        for criterion in criteria:
            if criterion.id in seen:
                raise ValueError(f"duplicate criterion id: {criterion.id}")
            seen.add(criterion.id)
        return criteria


def _load_text(source: str) -> tuple[str, str | None]:
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8"), path.suffix.lower()
    return source, None


def _parse_structured(text: str, suffix: str | None) -> dict[str, Any]:
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
    if source is None:
        return None
    if isinstance(source, Rubric):
        return source
    if isinstance(source, dict):
        return Rubric.model_validate(source)
    text, suffix = _load_text(str(source))
    data = _parse_structured(text, suffix)
    return Rubric.model_validate(data)


def _merge_weights(base: Criterion, override: Criterion) -> float:
    if override.weight != 1.0 and base.weight != 1.0:
        return base.weight * override.weight
    if override.weight != 1.0:
        return override.weight
    return base.weight


def blend_rubrics(base: Rubric | None, override: Rubric | None) -> Rubric | None:
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


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _score(
    criteria: Iterable[Criterion], values: dict[str, float], aggregation: str
) -> dict[str, Any]:
    if aggregation == "inherit":
        aggregation = "weighted_sum"
    per_criterion: dict[str, dict[str, Any]] = {}
    total = 0.0
    total_weight = 0.0
    for criterion in criteria:
        score = values.get(criterion.id, 0.0)
        per_criterion[criterion.id] = {
            "score": score,
            "weight": criterion.weight,
            "required": criterion.required,
        }
        if aggregation == "sum":
            total += score
        elif aggregation == "weighted_sum":
            total += score * criterion.weight
            total_weight += criterion.weight
    if aggregation == "weighted_sum" and total_weight > 0:
        total = total / total_weight
    if aggregation == "custom":
        total = None  # type: ignore[assignment]
    return {
        "aggregation": aggregation,
        "score": total,
        "per_criterion": per_criterion,
    }


def score_events_against_rubric(
    events: list[dict[str, Any]], rubric: Rubric | None
) -> dict[str, Any]:
    if rubric is None:
        return {"aggregation": "none", "score": None, "per_criterion": {}}
    values: dict[str, float] = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        cid = event.get("criterion_id") or event.get("id") or event.get("criterion")
        score = _as_float(event.get("score"))
        if cid and score is not None:
            values[str(cid)] = score
    return _score(rubric.criteria, values, rubric.aggregation)


def score_outcome_against_rubric(outcome: dict[str, Any], rubric: Rubric | None) -> dict[str, Any]:
    if rubric is None:
        return {"aggregation": "none", "score": None, "per_criterion": {}}
    values: dict[str, float] = {}
    if isinstance(outcome, dict):
        candidates = (
            outcome.get("criteria") if isinstance(outcome.get("criteria"), dict) else outcome
        )
        if isinstance(candidates, dict):
            for key, value in candidates.items():
                score = _as_float(value)
                if score is not None:
                    values[str(key)] = score
    return _score(rubric.criteria, values, rubric.aggregation)
