"""Dataset registry and helpers shared by Task Apps."""

from __future__ import annotations

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.localapi datasets") from exc

TaskDatasetSpec = getattr(synth_ai_py, "TaskDatasetSpec", dict)
TaskDatasetRegistry = getattr(synth_ai_py, "TaskDatasetRegistry", dict)

__all__ = ["TaskDatasetSpec", "TaskDatasetRegistry"]
