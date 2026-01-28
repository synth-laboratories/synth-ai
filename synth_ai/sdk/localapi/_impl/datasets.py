"""Dataset registry and helpers shared by Task Apps."""

from __future__ import annotations

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.localapi datasets") from exc

TaskDatasetSpec = synth_ai_py.TaskDatasetSpec
TaskDatasetRegistry = synth_ai_py.TaskDatasetRegistry

__all__ = ["TaskDatasetSpec", "TaskDatasetRegistry"]
