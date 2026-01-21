"""Registry for Task Apps exposed via the shared FastAPI harness.

Prefer synth_ai.sdk.localapi.apps moving forward. This module remains for
backward compatibility during the naming transition.
"""

from __future__ import annotations

import warnings

from synth_ai.sdk.localapi import apps as _localapi_apps
from synth_ai.sdk.localapi.apps import *  # noqa: F403

__all__ = list(_localapi_apps.__all__)

warnings.warn(
    "synth_ai.sdk.localapi._impl.apps is deprecated, use synth_ai.sdk.localapi.apps instead. "
    "Will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)
