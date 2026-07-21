"""Deprecated exact re-exports from the core Research implementation."""

from __future__ import annotations

import warnings

warnings.warn('synth_ai.managed_research is deprecated since synth-ai 0.16.0 and will be removed in 0.18.0 no earlier than 2026-09-01; use SynthClient().research.', DeprecationWarning, stacklevel=2)

from synth_ai.core.research._legacy import *  # noqa: F403
