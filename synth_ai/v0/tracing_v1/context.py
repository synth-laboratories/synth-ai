import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generic, Optional, TypeVar

from .local import (
    _local,
    active_events_var,
    system_id_var,
    system_instance_id_var,
    system_instance_metadata_var,
    system_name_var,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ContextState(Generic[T]):
    """Manages state that needs to be accessible in both sync and async contexts."""

    def __init__(self, thread_local_attr: str, context_var: T):
        self.thread_local_attr = thread_local_attr
        self.context_var = context_var

    def get(self) -> Optional[T]:
        """Get value from appropriate context."""
        try:
            asyncio.get_running_loop()
            return self.context_var.get() or getattr(_local, self.thread_local_attr, None)
        except RuntimeError:
            return getattr(_local, self.thread_local_attr, None)

    def set(self, value: T) -> Optional[T]:
        """Set value in appropriate context."""
        try:
            asyncio.get_running_loop()
            return self.context_var.set(value)
        except RuntimeError:
            setattr(_local, self.thread_local_attr, value)
            return None

    def reset(self, token: Optional[T] = None) -> None:
        """Reset/clear value from appropriate context."""
        try:
            asyncio.get_running_loop()
            if token is not None:
                self.context_var.reset(token)
            else:
                self.context_var.set(None)
        except RuntimeError:
            if hasattr(_local, self.thread_local_attr):
                delattr(_local, self.thread_local_attr)


# Create global context state managers
system_name_state = ContextState("system_name", system_name_var)
system_id_state = ContextState("system_id", system_id_var)
system_instance_id_state = ContextState("system_instance_id", system_instance_id_var)
system_instance_metadata_state = ContextState(
    "system_instance_metadata", system_instance_metadata_var
)
active_events_state = ContextState("active_events", active_events_var)


@contextmanager
def trace_context(
    system_name: str,
    system_id: str,
    system_instance_id: str,
    system_instance_metadata: Optional[Dict[str, Any]] = None,
):
    """Context manager for setting up tracing context.

    This ensures proper setup and cleanup of context variables in both sync and async code.
    Also handles nested decorators by preserving existing context if present.

    Args:
        system_name: Name of the system
        system_id: ID of the system
        system_instance_id: Instance ID of the system
        system_instance_metadata: Optional metadata for the system instance
    """
    # Store existing context if any
    prev_system_name = system_name_state.get()
    prev_system_id = system_id_state.get()
    prev_system_instance_id = system_instance_id_state.get()
    prev_system_instance_metadata = system_instance_metadata_state.get()
    prev_active_events = active_events_state.get()

    try:
        # Set new context
        system_name_token = system_name_state.set(system_name)
        system_id_token = system_id_state.set(system_id)
        system_instance_id_token = system_instance_id_state.set(system_instance_id)

        # Set system_instance_metadata if provided, otherwise use empty dict
        metadata_to_set = system_instance_metadata if system_instance_metadata is not None else {}
        system_instance_metadata_token = system_instance_metadata_state.set(metadata_to_set)

        # Initialize active events if not present
        if not active_events_state.get():
            active_events_state.set({})

        yield
    finally:
        # Restore previous context if it existed
        if prev_system_name is not None:
            system_name_state.set(prev_system_name)
        else:
            system_name_state.reset(system_name_token)

        if prev_system_id is not None:
            system_id_state.set(prev_system_id)
        else:
            system_id_state.reset(system_id_token)

        if prev_system_instance_id is not None:
            system_instance_id_state.set(prev_system_instance_id)
        else:
            system_instance_id_state.reset(system_instance_id_token)

        if prev_system_instance_metadata is not None:
            system_instance_metadata_state.set(prev_system_instance_metadata)
        else:
            system_instance_metadata_state.reset(system_instance_metadata_token)

        if prev_active_events is not None:
            active_events_state.set(prev_active_events)
        else:
            active_events_state.reset()


def get_current_context() -> Dict[str, Any]:
    """Get the current tracing context.

    Returns:
        Dict containing current system_name, system_id, system_instance_id, and system_instance_metadata
    """
    return {
        "system_name": system_name_state.get(),
        "system_id": system_id_state.get(),
        "system_instance_id": system_instance_id_state.get(),
        "system_instance_metadata": system_instance_metadata_state.get() or {},
    }
