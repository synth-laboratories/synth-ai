"""Compatibility re-export; implementation lives in contracts.smr_inference_providers."""

from synth_ai.core.research.contracts.smr_inference_providers import (
    SMR_INFERENCE_PROVIDER_VALUES,
    SmrInferenceProvider,
    coerce_smr_inference_provider,
)

__all__ = [
    'SMR_INFERENCE_PROVIDER_VALUES',
    'SmrInferenceProvider',
    'coerce_smr_inference_provider',
]
