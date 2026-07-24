"""Compatibility re-export; implementation lives in contracts.smr_run_policy."""

from synth_ai.core.research.contracts.smr_run_policy import (
    SmrRunPolicy,
    SmrRunPolicyAccess,
    SmrRunPolicyLimits,
    coerce_smr_run_policy,
    coerce_smr_run_policy_access,
    coerce_smr_run_policy_limits,
)

__all__ = [
    'SmrRunPolicy',
    'SmrRunPolicyAccess',
    'SmrRunPolicyLimits',
    'coerce_smr_run_policy',
    'coerce_smr_run_policy_access',
    'coerce_smr_run_policy_limits',
]
