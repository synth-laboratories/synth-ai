#!/usr/bin/env python3
"""Compatibility wrapper around ``synth_ai.sdk.managed_research``."""

from synth_ai.sdk.managed_research import (
    ACTIVE_RUN_STATES,
    SmrApiError,
    SmrControlClient,
    first_id,
)

__all__ = ["ACTIVE_RUN_STATES", "SmrApiError", "SmrControlClient", "first_id"]
