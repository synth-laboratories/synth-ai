"""Hook system for extending tracing functionality.

The hook system provides a flexible way to extend the tracing system without
modifying core code. Hooks can be registered at various points in the tracing
lifecycle to perform custom processing, validation, or enrichment.

Hook Points:
-----------
- session_start: Called when a new session begins
- session_end: Called when a session ends
- timestep_start: Called when a new timestep begins
- timestep_end: Called when a timestep completes
- event_recorded: Called after an event is recorded
- message_recorded: Called after a message is recorded
- before_save: Called before saving a session to database
- after_save: Called after successful save to database

Hook Design:
-----------
Hooks are designed to be:
1. **Non-blocking**: Hooks should execute quickly to avoid impacting performance
2. **Fault-tolerant**: Hook failures don't stop the tracing pipeline
3. **Prioritized**: Hooks execute in priority order (highest first)
4. **Async-aware**: Support both sync and async callbacks

Common Use Cases:
----------------
- Data validation before storage
- Metric calculation and aggregation
- External system notifications
- Data enrichment and transformation
- Custom filtering and sampling
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .abstractions import BaseEvent


@dataclass
class Hook:
    """A hook that can be registered with the tracer.

    Attributes:
        name: Unique identifier for the hook
        callback: Function to call when hook is triggered. Can be sync or async.
        event_types: Optional list of event types to filter on (for event_recorded hook)
        priority: Execution priority (higher numbers run first)
        enabled: Whether the hook is currently active
    """

    name: str
    callback: Callable
    event_types: list[str] | None = None
    priority: int = 0
    enabled: bool = True


class HookManager:
    """Manages hooks for session tracing.

    The HookManager maintains collections of hooks for each hook point and
    handles their execution. It ensures hooks are called in priority order
    and handles both sync and async callbacks appropriately.

    Thread Safety:
        The HookManager is designed to be thread-safe for registration and
        execution. Multiple async tasks can trigger hooks concurrently.
    """

    def __init__(self):
        self.hooks: dict[str, list[Hook]] = {
            "session_start": [],
            "session_end": [],
            "timestep_start": [],
            "timestep_end": [],
            "event_recorded": [],
            "message_recorded": [],
            "before_save": [],
            "after_save": [],
        }

    def register(
        self,
        event: str,
        callback: Callable,
        name: str | None = None,
        priority: int = 0,
        event_types: list[str] | None = None,
    ) -> Hook:
        """Register a new hook.

        Args:
            event: Hook point name (e.g., 'session_start', 'event_recorded')
            callback: Function to call. Signature depends on hook point:
                     - session_start(session_id: str, metadata: Dict)
                     - event_recorded(event_obj: BaseEvent)
                     - etc. (see individual hook point docs)
            name: Optional name for the hook (defaults to callback.__name__)
            priority: Execution priority (higher = earlier execution)
            event_types: For 'event_recorded' hook, filter to specific event types

        Returns:
            The created Hook instance

        Raises:
            ValueError: If the event name is not a valid hook point
        """
        if event not in self.hooks:
            raise ValueError(f"Unknown hook event: {event}")

        hook = Hook(
            name=name or getattr(callback, "__name__", "unknown"),
            callback=callback,
            event_types=event_types,
            priority=priority,
        )

        self.hooks[event].append(hook)
        self.hooks[event].sort(key=lambda h: h.priority, reverse=True)

        return hook

    def unregister(self, event: str, name: str):
        """Unregister a hook by name.

        Args:
            event: Hook point name
            name: Name of the hook to remove
        """
        if event not in self.hooks:
            return

        self.hooks[event] = [h for h in self.hooks[event] if h.name != name]

    async def trigger(self, event: str, *args, **kwargs) -> list[Any]:
        """Trigger all hooks for an event.

        Executes all registered hooks for the given event in priority order.
        Handles both sync and async callbacks appropriately. Exceptions in
        hooks are caught and logged but don't stop execution of other hooks.

        Args:
            event: Hook point name
            *args: Positional arguments passed to hook callbacks
            **kwargs: Keyword arguments passed to hook callbacks

        Returns:
            List of return values from all executed hooks
        """
        if event not in self.hooks:
            return []

        results = []
        for hook in self.hooks[event]:
            if not hook.enabled:
                continue

            # Check event type filter - this allows hooks to only process
            # specific types of events (e.g., only LMCAISEvent)
            if hook.event_types and "event_obj" in kwargs:
                event_obj = kwargs["event_obj"]
                if (
                    hasattr(event_obj, "event_type")
                    and event_obj.event_type not in hook.event_types
                ):
                    continue

            try:
                # Handle both async and sync callbacks transparently
                if asyncio.iscoroutinefunction(hook.callback):
                    result = await hook.callback(*args, **kwargs)
                else:
                    result = hook.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but don't stop processing - hooks should not
                # break the main tracing pipeline
                print(f"Hook {hook.name} failed: {e}")

        return results


# Default hooks for common use cases
def create_default_hooks() -> HookManager:
    """Create hook manager with default hooks.

    Sets up a basic set of hooks that provide common functionality:
    - Session start logging
    - Event validation
    - Automatic event enrichment

    Returns:
        HookManager with default hooks registered
    """
    manager = HookManager()

    # Example: Log session starts - useful for debugging and monitoring
    async def log_session_start(session_id: str, metadata: dict[str, Any]):
        import os

        if os.getenv("SYNTH_TRACE_VERBOSE", "0") in ("1", "true", "True"):
            print(f"Session started: {session_id}")

    # Example: Validate events before recording - ensures data quality
    def validate_event(event_obj: BaseEvent) -> bool:
        if not event_obj.system_instance_id:
            raise ValueError("Event must have system_instance_id")
        return True

    # Example: Add metadata to all events - useful for versioning and tracking
    def enrich_event(event_obj: BaseEvent):
        if "enriched" not in event_obj.metadata:
            event_obj.metadata["enriched"] = True
            event_obj.metadata["hook_version"] = "1.0"

    # Register default hooks
    manager.register("session_start", log_session_start, priority=10)
    manager.register("event_recorded", validate_event, name="validate_event", priority=20)
    manager.register("event_recorded", enrich_event, name="enrich_event", priority=15)

    return manager


# Global hooks instance - applications can customize this or create their own
# HookManager instances for different use cases
GLOBAL_HOOKS = create_default_hooks()
