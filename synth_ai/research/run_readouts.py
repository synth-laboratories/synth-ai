"""Deprecated module — the public noun is Swarm; import from ``synth_ai.research.swarm_readouts``."""

from __future__ import annotations

import warnings

from synth_ai.research.swarm_readouts import (
    ResearchSwarmArtifactsAPI as ResearchRunArtifactsAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmEventsAPI as ResearchRunEventsAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmLogsAPI as ResearchRunLogsAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmMessageQueueAPI as ResearchRunMessageQueueAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmMilestonesAPI as ResearchRunMilestonesAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmOrchestratorAPI as ResearchRunOrchestratorAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmProgressAPI as ResearchRunProgressAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmReadoutsMixin as ResearchRunReadoutsMixin,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmResultsAPI as ResearchRunResultsAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmSnapshotsAPI as ResearchRunSnapshotsAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmTasksAPI as ResearchRunTasksAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmTranscriptAPI as ResearchRunTranscriptAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmUsageAPI as ResearchRunUsageAPI,
)
from synth_ai.research.swarm_readouts import (
    ResearchSwarmWorkProductsAPI as ResearchRunWorkProductsAPI,
)
from synth_ai.research.swarm_readouts import _deprecated_method

warnings.warn(
    "synth_ai.research.run_readouts is deprecated; import from "
    "synth_ai.research.swarm_readouts (Managed Swarm noun) instead.",
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str) -> object:
    """Map any legacy ``ResearchRun*`` deep class to its ``ResearchSwarm*`` original."""
    if name.startswith("ResearchRun"):
        from synth_ai.research import swarm_readouts

        target = "ResearchSwarm" + name[len("ResearchRun") :]
        try:
            return getattr(swarm_readouts, target)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ResearchRunReadoutsMixin",
    "ResearchRunUsageAPI",
    "ResearchRunProgressAPI",
    "ResearchRunSnapshotsAPI",
    "ResearchRunEventsAPI",
    "ResearchRunTasksAPI",
    "ResearchRunMessageQueueAPI",
    "ResearchRunTranscriptAPI",
    "ResearchRunWorkProductsAPI",
    "ResearchRunArtifactsAPI",
    "ResearchRunResultsAPI",
    "ResearchRunLogsAPI",
    "ResearchRunMilestonesAPI",
    "ResearchRunOrchestratorAPI",
    "_deprecated_method",
]
