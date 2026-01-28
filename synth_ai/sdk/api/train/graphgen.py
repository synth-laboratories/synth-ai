"""Backward-compatible GraphGen training exports."""

from __future__ import annotations

from synth_ai.sdk.optimization.internal.graphgen import (  # noqa: F401
    GraphEvolveJob as GraphGenJob,
)
from synth_ai.sdk.optimization.internal.graphgen import (
    GraphEvolveJobResult as GraphGenJobResult,
)
from synth_ai.sdk.optimization.internal.graphgen import (
    GraphEvolveSubmitResult as GraphGenSubmitResult,
)
from synth_ai.sdk.optimization.internal.graphgen_models import (  # noqa: F401
    load_graphgen_taskset,
    parse_graphgen_taskset,
)

__all__ = [
    "GraphGenJob",
    "GraphGenJobResult",
    "GraphGenSubmitResult",
    "load_graphgen_taskset",
    "parse_graphgen_taskset",
]
