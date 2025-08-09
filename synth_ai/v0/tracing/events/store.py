import json
import logging
import time
from threading import RLock  # Change this import
from typing import Any, Dict, List

from ..abstractions import Event, EventPartitionElement, SystemTrace
from ..local import (  # Import context variables
    _local,
    active_events_var,
)

logger = logging.getLogger(__name__)


class EventStore:
    def __init__(self) -> None:
        self._traces: Dict[str, SystemTrace] = {}
        self._lock = RLock()  # Use RLock instead of Lock
        self.logger = logging.getLogger(__name__)

    def get_or_create_system_trace(
        self,
        system_name: str,
        system_id: str,
        system_instance_id: str,
        _already_locked: bool = False,
    ) -> SystemTrace:
        """Get or create a SystemTrace for the given system_instance_id."""
        logger = logging.getLogger(__name__)
        # logger.debug(f"Starting get_or_create_system_trace for {system_instance_id}")

        def _get_or_create():
            # logger.debug("Inside _get_or_create")
            if system_instance_id not in self._traces:
                # Get system_instance_metadata from context if available
                system_instance_metadata = {}
                try:
                    from ..context import get_current_context

                    context = get_current_context()
                    if (
                        "system_instance_metadata" in context
                        and context["system_instance_metadata"]
                    ):
                        system_instance_metadata = context["system_instance_metadata"]
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get system_instance_metadata from context: {str(e)}"
                    )

                # logger.debug(f"Creating new system trace for {system_instance_id}")
                self._traces[system_instance_id] = SystemTrace(
                    system_name=system_name,
                    system_id=system_id,
                    system_instance_id=system_instance_id,
                    metadata={},
                    partition=[],  # EventPartitionElement(partition_index=0, events=[])
                    current_partition_index=0,
                    instance_metadata=system_instance_metadata,
                )
            # logger.debug("Returning system trace")
            return self._traces[system_instance_id]

        if _already_locked:
            return _get_or_create()
        else:
            with self._lock:
                # logger.debug("Lock acquired in get_or_create_system_trace")
                return _get_or_create()

    def increment_partition(self, system_name: str, system_id: str, system_instance_id: str) -> int:
        """Increment the partition index for a system and create new partition element."""
        logger = logging.getLogger(__name__)
        # logger.debug(f"Starting increment_partition for system {system_instance_id}")

        with self._lock:
            # logger.debug("Lock acquired in increment_partition")
            system_trace = self.get_or_create_system_trace(
                system_name, system_id, system_instance_id, _already_locked=True
            )
            # logger.debug(
            #     f"Got system trace, current index: {system_trace.current_partition_index}"
            # )

            system_trace.current_partition_index += 1
            # logger.debug(
            #     f"Incremented index to: {system_trace.current_partition_index}"
            # )

            system_trace.partition.append(
                EventPartitionElement(
                    partition_index=system_trace.current_partition_index, events=[]
                )
            )
            # logger.debug("Added new partition element")

            return system_trace.current_partition_index

    def add_event(self, system_name: str, system_id: str, system_instance_id: str, event: Event):
        """Add an event to the appropriate partition of the system trace."""
        # self.#logger.debug(f"Adding event type {event.event_type} to system {system_instance_id}")
        # self.#logger.debug(
        #     f"Event details: opened={event.opened}, closed={event.closed}, partition={event.partition_index}"
        # )
        # print("Adding event: ", event)

        # try:
        if not self._lock.acquire(timeout=5):
            self.logger.error("Failed to acquire lock within timeout period")
            return

        try:
            system_trace = self.get_or_create_system_trace(
                system_name, system_id, system_instance_id, _already_locked=True
            )
            # self.#logger.debug(
            #     f"Got system trace with {len(system_trace.partition)} partitions"
            # )

            current_partition = next(
                (p for p in system_trace.partition if p.partition_index == event.partition_index),
                None,
            )

            if current_partition is None:
                self.logger.error(
                    f"No partition found for index {event.partition_index} - existing partitions: {set([p.partition_index for p in system_trace.partition])}"
                )
                raise ValueError(f"No partition found for index {event.partition_index}")

            current_partition.events.append(event)
        finally:
            self._lock.release()
        # except Exception as e:
        #     self.logger.error(f"Error in add_event: {str(e)}", exc_info=True)
        #     raise

    def get_system_traces(self) -> List[SystemTrace]:
        """Get all system traces."""
        with self._lock:
            self.end_all_active_events()

            return list(self._traces.values())

    def end_all_active_events(self):
        """End all active events and store them."""
        # self.#logger.debug("Ending all active events")

        # For synchronous code
        if hasattr(_local, "active_events"):
            for _, event in list(_local.active_events.items()):
                if event.closed is None:
                    event.closed = time.time()
                    self.add_event(
                        event.system_name,
                        event.system_id,
                        event.system_instance_id,
                        event,
                    )
            _local.active_events.clear()

        else:
            # For asynchronous code
            active_events_async = active_events_var.get()
            if active_events_async:
                for _, event in list(active_events_async.items()):
                    if event.closed is None:
                        event.closed = time.time()
                        self.add_event(
                            event.system_name,
                            event.system_id,
                            event.system_instance_id,
                            event,
                        )
                active_events_var.set({})

    def get_system_traces_json(self) -> str:
        """Get all system traces as JSON."""
        with self._lock:
            return json.dumps(
                [
                    {
                        "system_instance_id": trace.system_instance_id,
                        "current_partition_index": trace.current_partition_index,
                        "partition": [
                            {
                                "partition_index": p.partition_index,
                                "events": [self._event_to_dict(event) for event in p.events],
                            }
                            for p in trace.partition
                        ],
                    }
                    for trace in self._traces.values()
                ],
                default=str,
            )

    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert an Event object to a dictionary."""
        return {
            "event_type": event.event_type,
            "opened": event.opened,
            "closed": event.closed,
            "partition_index": event.partition_index,
            "agent_compute_step": {
                "event_order": event.agent_compute_step.event_order,
                "compute_began": event.agent_compute_step.compute_began,
                "compute_ended": event.agent_compute_step.compute_ended,
                "compute_input": event.agent_compute_step.compute_input,
                "compute_output": event.agent_compute_step.compute_output,
            },
            "environment_compute_steps": [
                {
                    "event_order": step.event_order,
                    "compute_began": step.compute_began,
                    "compute_ended": step.compute_ended,
                    "compute_input": step.compute_input,
                    "compute_output": step.compute_output,
                }
                for step in event.environment_compute_steps
            ],
        }


# Global event store instance
event_store = EventStore()
