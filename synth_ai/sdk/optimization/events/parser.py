"""Event parser for converting raw backend events to typed OpenResponses events.

DEPRECATED: This module has moved to synth_ai.sdk.shared.orchestration.events.parser.
This re-export is provided for backwards compatibility.
"""

from __future__ import annotations

# Re-export everything from the new canonical location
from synth_ai.sdk.shared.orchestration.events.parser import (
    get_event_type,
    is_failure_event,
    is_success_event,
    is_terminal_event,
    parse_event,
)

__all__ = [
    "parse_event",
    "is_terminal_event",
    "is_success_event",
    "is_failure_event",
    "get_event_type",
]
