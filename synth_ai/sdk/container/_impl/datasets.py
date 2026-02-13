"""Dataset registry and helpers shared by Task Apps."""

from __future__ import annotations

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None

if synth_ai_py is None:
    TaskDatasetSpec = dict
    TaskDatasetRegistry = dict
else:
    TaskDatasetSpec = getattr(synth_ai_py, "TaskDatasetSpec", dict)
    TaskDatasetRegistry = getattr(synth_ai_py, "TaskDatasetRegistry", dict)

__all__ = ["TaskDatasetSpec", "TaskDatasetRegistry"]
