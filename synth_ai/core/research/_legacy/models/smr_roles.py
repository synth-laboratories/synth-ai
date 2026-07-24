"""Compatibility re-export; implementation lives in contracts.smr_roles."""

from synth_ai.core.research.contracts.smr_roles import (
    RoleBinding,
    RoleProviderRequirement,
    SmrRoleBindings,
    WorkerRolePalette,
    coerce_smr_role_bindings,
)

__all__ = [
    'RoleBinding',
    'RoleProviderRequirement',
    'SmrRoleBindings',
    'WorkerRolePalette',
    'coerce_smr_role_bindings',
]
