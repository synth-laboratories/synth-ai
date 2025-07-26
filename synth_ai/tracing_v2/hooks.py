"""
Hook classes for state analysis in the tracing system.
"""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

from .abstractions import SessionEvent, EventMetadata


# Hook classes for state analysis
@dataclass
class HookResult(EventMetadata):
    """Result from a hook execution."""
    hook_name: str = ""
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    code: str = "?"  # Single-letter code for TUI display
    priority: int = 0  # Priority for display (higher = more important)


class TraceHook:
    """Base class for all trace hooks."""
    def check(self, event: SessionEvent) -> Optional[HookResult]:
        """Check if this hook should fire for the given event."""
        raise NotImplementedError


class TraceStateHook(TraceHook):
    """Hook that analyzes state transitions in events."""
    def check(self, event: SessionEvent) -> Optional[HookResult]:
        """Check state before/after in the event."""
        if hasattr(event, 'system_state_before') and hasattr(event, 'system_state_after'):
            return self.analyze_state(event.system_state_before, event.system_state_after, event)
        return None
    
    def analyze_state(self, state_before: Any, state_after: Any, event: SessionEvent) -> Optional[HookResult]:
        """Analyze state transition. Override in subclasses."""
        raise NotImplementedError


# Alias for backward compatibility
HookMetadataResult = HookResult