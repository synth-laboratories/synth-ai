"""Centralized enums for Synth AI SDK.

Rust-backed enum values for consistency across SDKs.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.enums.") from exc


def _load_enum_values_fallback() -> Dict[str, Dict[str, str]]:
    asset = (
        Path(__file__).resolve().parents[2] / "synth_ai_core" / "assets" / "data_enum_values.json"
    )
    if asset.exists():
        return json.loads(asset.read_text())
    return {}


if synth_ai_py is None or not hasattr(synth_ai_py, "data_enum_values"):
    _ENUM_VALUES = _load_enum_values_fallback()
else:
    _ENUM_VALUES = synth_ai_py.data_enum_values()


class _StrEnum(str, Enum):
    """Base class for string enums."""

    pass


def _build_enum(name: str):
    return Enum(name, _ENUM_VALUES.get(name, {}), type=_StrEnum, module=__name__)


JobType = _build_enum("JobType")
JobStatus = _build_enum("JobStatus")
InferenceMode = _build_enum("InferenceMode")
ProviderName = _build_enum("ProviderName")
RewardSource = _build_enum("RewardSource")
RewardType = _build_enum("RewardType")
RewardScope = _build_enum("RewardScope")
ObjectiveKey = _build_enum("ObjectiveKey")
ObjectiveDirection = _build_enum("ObjectiveDirection")
GraphType = _build_enum("GraphType")
OptimizationMode = _build_enum("OptimizationMode")
VerifierMode = _build_enum("VerifierMode")
TrainingType = _build_enum("TrainingType")
AdaptiveCurriculumLevel = _build_enum("AdaptiveCurriculumLevel")
AdaptiveBatchLevel = _build_enum("AdaptiveBatchLevel")
SynthModelName = _build_enum("SynthModelName")
SuccessStatus = _build_enum("SuccessStatus")
OutputMode = _build_enum("OutputMode")

__all__ = [
    "JobType",
    "JobStatus",
    "InferenceMode",
    "ProviderName",
    "RewardSource",
    "RewardType",
    "RewardScope",
    "ObjectiveKey",
    "ObjectiveDirection",
    "GraphType",
    "OptimizationMode",
    "VerifierMode",
    "TrainingType",
    "AdaptiveCurriculumLevel",
    "AdaptiveBatchLevel",
    "SynthModelName",
    "SuccessStatus",
    "OutputMode",
]
