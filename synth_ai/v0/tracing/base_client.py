import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .abstractions import Event
from .config import TracingConfig
from .events.store import event_store

logger = logging.getLogger(__name__)


@dataclass
class LogResponse:
    """Represents the response from a logging attempt"""

    success: bool
    error: Optional[str] = None
    retry_after: Optional[float] = None
    status_code: Optional[int] = None


class BaseLogClient(ABC):
    """Abstract base class for logging clients"""

    def __init__(self, config: TracingConfig):
        self.config = config
        self._consecutive_failures = 0
        self._last_failure_time = 0
        self._circuit_open = False
        self._circuit_open_time = 0

    def _should_retry(self, attempt: int, status_code: Optional[int] = None) -> bool:
        """Determine if a retry should be attempted based on configuration and status"""
        if attempt >= self.config.max_retries:
            return False

        # Don't retry on certain status codes
        if status_code:
            # Always retry on 429 (Too Many Requests)
            if status_code == 429:
                return True

            # Don't retry on client errors except timeout/too many requests
            if 400 <= status_code < 500 and status_code not in (408, 429):
                return False

        return True

    def _prepare_payload(self, event: Event, system_info: Dict[str, str]) -> Dict[str, Any]:
        """Prepare the payload for sending"""
        return {
            "event": event.to_dict(),
            "system_info": system_info,
            "timestamp": time.time(),
            "sdk_version": self.config.sdk_version,  # Use SDK version from config
        }

    def _handle_failure(self, event: Event, system_info: Dict[str, str], error: Exception) -> None:
        """Handle logging failure by storing in event_store"""
        logger.error(f"Logging failed: {str(error)}")
        self._consecutive_failures += 1
        self._last_failure_time = time.time()

        # Store in event_store as backup
        event_store.add_event(
            system_info["system_name"],
            system_info["system_id"],
            system_info["system_instance_id"],
            event,
        )

    def _handle_success(self) -> None:
        """Handle successful logging attempt"""
        self._consecutive_failures = 0
        self._last_failure_time = 0

    @abstractmethod
    def send_event(self, event: Event, system_info: Dict[str, str]) -> bool:
        """Send a single event with retries and fallback"""
        pass


class BaseAsyncLogClient(BaseLogClient):
    """Abstract base class for async logging clients"""

    @abstractmethod
    async def send_event(self, event: Event, system_info: Dict[str, str]) -> bool:
        """Send a single event with retries and fallback (async version)"""
        pass
