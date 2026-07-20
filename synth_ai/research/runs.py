"""Deprecated module — the public noun is Swarm; import from ``synth_ai.research.swarms``.

``research.runs`` remains as a compatibility alias for one release cycle.
"""

from __future__ import annotations

import warnings

from synth_ai.research.swarms import (
    ResearchSwarmHandle as ResearchRunHandle,
)
from synth_ai.research.swarms import (
    ResearchSwarmHandle as ResearchRunSession,
)
from synth_ai.research.swarms import (
    ResearchSwarmsAPI as ResearchRunsAPI,
)

warnings.warn(
    "synth_ai.research.runs is deprecated; import from synth_ai.research.swarms "
    "(Managed Swarm noun) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ResearchRunHandle", "ResearchRunSession", "ResearchRunsAPI"]
