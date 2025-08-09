import time
from contextlib import contextmanager

from ..abstractions import Event
from ..config import LoggingMode
from ..decorators import (
    _local,
    clear_current_event,
    get_tracing_config,
    set_current_event,
)
from ..events.store import event_store
from ..immediate_client import (
    AsyncImmediateLogClient,
    ImmediateLogClient,
)
from ..local import (
    system_id_var,
    system_instance_id_var,
    system_name_var,
)


@contextmanager
def event_scope(event_type: str):
    """
    Context manager for creating and managing events.

    Usage:
        with event_scope("my_event_type"):
            # do stuff
    """
    # Check if we're in an async context
    try:
        import asyncio

        asyncio.get_running_loop()
        is_async = True
    except RuntimeError:
        is_async = False

    # Get system_instance_id from appropriate source
    system_instance_id = (
        system_instance_id_var.get() if is_async else getattr(_local, "system_instance_id", None)
    )
    system_id = system_id_var.get() if is_async else getattr(_local, "system_id", None)
    system_name = system_name_var.get() if is_async else getattr(_local, "system_name", None)

    event = Event(
        system_instance_id=system_instance_id,
        event_type=event_type,
        opened=time.time(),
        closed=None,
        partition_index=0,
        agent_compute_step=None,
        environment_compute_steps=[],
    )
    set_current_event(event)

    try:
        yield event
    finally:
        event.closed = time.time()
        clear_current_event(event_type)

        # Get the config to determine logging mode
        config = get_tracing_config()

        # If immediate logging is enabled and we have system info, send the event now
        if config.mode == LoggingMode.INSTANT and system_instance_id:
            system_info = {
                "system_name": system_name,
                "system_id": system_id,
                "system_instance_id": system_instance_id,
            }
            if is_async:
                client = AsyncImmediateLogClient(config)
                # Note: Since we can't use await in a finally block,
                # we'll have to rely on the event store as primary storage in async context
            else:
                client = ImmediateLogClient(config)
                client.send_event(event, system_info)

        # Always store in event_store
        if system_instance_id:
            event_store.add_event(system_name, system_id, system_instance_id, event)
