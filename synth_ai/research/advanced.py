"""Advanced Research compatibility surface for operator and eval integrations.

This module is public but unstable.  It keeps active operator integrations off
the deprecated :mod:`synth_ai.managed_research` package while the corresponding
capabilities graduate into the typed ``projects``, ``swarms``, ``factories``,
``evidence``, and ``resources`` namespaces.  New customer code should use
``SynthClient().research`` instead.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from synth_ai.core.research._legacy import *  # noqa: F403
from synth_ai.core.research._legacy.models.local_execution_profile import (
    build_local_launch_payload,
    default_local_eval_contract_path,
    load_local_eval_contract,
    load_local_execution_profile,
    load_local_execution_profiles,
    local_execution_payload,
)
from synth_ai.core.research._legacy.models.run_control import (
    ManagedResearchRunControlEnqueueStatus,
    ManagedResearchRunControlError,
    RunLifecycleControlErrorCode,
)
from synth_ai.core.research._legacy.models.run_state import ManagedResearchRun
from synth_ai.core.research._legacy.models.smr_runbooks import (
    SMR_RUNBOOK_KIND_VALUES,
    SmrRunbookKind,
)
from synth_ai.core.research._legacy.models.smr_work_modes import SMR_WORK_MODE_VALUES
from synth_ai.core.research._legacy.models.types import KickoffContract
from synth_ai.core.research._legacy.sdk.runs import RunHandle
from synth_ai.core.research._legacy.version import __version__
