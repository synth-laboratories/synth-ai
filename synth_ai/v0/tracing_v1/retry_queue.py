import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .abstractions import Event
from .config import TracingConfig

logger = logging.getLogger(__name__)


@dataclass
class QueuedEvent:
    """Represents an event that failed to upload and needs to be retried."""

    event: Event
    system_info: Dict[str, str]
    attempt_count: int = 0
    last_attempt: float = 0


class RetryQueue:
    """Manages failed event uploads with retry capabilities."""

    def __init__(self, config: TracingConfig):
        self.config = config
        self.queue: deque[QueuedEvent] = deque()
        self._lock = threading.Lock()
        self._is_processing = False
        self._batch_size = config.batch_size

    def add_failed_event(self, event: Event, system_info: Dict[str, str]) -> None:
        """Add a failed event to the retry queue."""
        with self._lock:
            # Check if event is already in queue to avoid duplicates
            for queued in self.queue:
                if (
                    queued.event.system_instance_id == event.system_instance_id
                    and queued.event.event_type == event.event_type
                    and queued.event.opened == event.opened
                ):
                    return

            self.queue.append(
                QueuedEvent(
                    event=event,
                    system_info=system_info,
                    attempt_count=0,
                    last_attempt=time.time(),
                )
            )
            logger.debug(f"Added event to retry queue. Queue size: {len(self.queue)}")

    def get_retryable_events(self, max_events: Optional[int] = None) -> List[QueuedEvent]:
        """Get events that are ready to be retried."""
        now = time.time()
        retryable = []

        with self._lock:
            for _ in range(len(self.queue)):
                if max_events and len(retryable) >= max_events:
                    break

                event = self.queue[0]
                # Use exponential backoff with the configured multiplier
                backoff = self.config.retry_backoff**event.attempt_count
                if now - event.last_attempt >= backoff:
                    retryable.append(self.queue.popleft())
                else:
                    # If this event isn't ready, later ones won't be either
                    break

        return retryable

    def process_sync(self) -> Tuple[int, int]:
        """Process the retry queue synchronously.

        Returns:
            Tuple of (success_count, failure_count)
        """
        if self._is_processing:
            return 0, 0

        self._is_processing = True
        success_count = 0
        failure_count = 0

        try:
            from .immediate_client import (
                ImmediateLogClient,  # Import here to avoid circular import
            )

            client = ImmediateLogClient(self.config)

            while True:
                batch = self.get_retryable_events(self._batch_size)
                if not batch:
                    break

                for queued_event in batch:
                    try:
                        if client.send_event(queued_event.event, queued_event.system_info):
                            success_count += 1
                            logger.debug(
                                f"Successfully retried event: {queued_event.event.event_type}"
                            )
                        else:
                            failure_count += 1
                            queued_event.attempt_count += 1
                            queued_event.last_attempt = time.time()
                            if queued_event.attempt_count < self.config.max_retries:
                                self.add_failed_event(queued_event.event, queued_event.system_info)
                            else:
                                logger.error(
                                    f"Event exhausted retry attempts: {queued_event.event.event_type}"
                                )
                    except Exception as e:
                        logger.error(f"Error processing retry queue: {e}")
                        failure_count += 1

        finally:
            self._is_processing = False

        return success_count, failure_count

    async def process_async(self) -> Tuple[int, int]:
        """Process the retry queue asynchronously.

        Returns:
            Tuple of (success_count, failure_count)
        """
        if self._is_processing:
            return 0, 0

        self._is_processing = True
        success_count = 0
        failure_count = 0

        try:
            from synth_ai.tracing_v1.immediate_client import (
                AsyncImmediateLogClient,  # Import here to avoid circular import
            )

            client = AsyncImmediateLogClient(self.config)

            while True:
                batch = self.get_retryable_events(self._batch_size)
                if not batch:
                    break

                for queued_event in batch:
                    try:
                        if await client.send_event(queued_event.event, queued_event.system_info):
                            success_count += 1
                            logger.debug(
                                f"Successfully retried event: {queued_event.event.event_type}"
                            )
                        else:
                            failure_count += 1
                            queued_event.attempt_count += 1
                            queued_event.last_attempt = time.time()
                            if queued_event.attempt_count < self.config.max_retries:
                                self.add_failed_event(queued_event.event, queued_event.system_info)
                            else:
                                logger.error(
                                    f"Event exhausted retry attempts: {queued_event.event.event_type}"
                                )
                    except Exception as e:
                        logger.error(f"Error processing retry queue: {e}")
                        failure_count += 1

        finally:
            self._is_processing = False

        return success_count, failure_count


# Global retry queue instance
retry_queue = RetryQueue(TracingConfig(api_key=""))  # Will be initialized with proper config later


def initialize_retry_queue(config: TracingConfig) -> None:
    """Initialize the global retry queue with the given config."""
    global retry_queue
    retry_queue = RetryQueue(config)
