"""Registry for Containers exposed via the shared FastAPI harness.

Prefer synth_ai.sdk.container.apps moving forward. This module remains for
backward compatibility during the naming transition.
"""

from __future__ import annotations

import warnings

from synth_ai.sdk.container import apps as _container_apps
from synth_ai.sdk.container.apps import *  # noqa: F403

__all__ = list(_container_apps.__all__)

warnings.warn(
    "synth_ai.sdk.container._impl.apps is deprecated, use synth_ai.sdk.container.apps instead. "
    "Will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)
