"""Compatibility re-export; implementation lives in contracts.smr_credential_providers."""

from synth_ai.core.research.contracts.smr_credential_providers import (
    SMR_CREDENTIAL_PROVIDER_VALUES,
    SmrCredentialProvider,
    coerce_smr_credential_provider,
)

__all__ = [
    'SMR_CREDENTIAL_PROVIDER_VALUES',
    'SmrCredentialProvider',
    'coerce_smr_credential_provider',
]
