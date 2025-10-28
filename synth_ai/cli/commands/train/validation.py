from __future__ import annotations

from synth_ai.api.train.env_resolver import resolve_env

__all__ = ["validate_train_environment"]


def validate_train_environment(*args, **kwargs):
    """Validate and resolve environment secrets used by the train command."""
    return resolve_env(*args, **kwargs)
