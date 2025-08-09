from typing import Dict

from .abstractions import Event
from .config import TracingConfig


class BaseLogClient:
    """Base class for synchronous logging clients"""

    def __init__(self, config: TracingConfig):
        self.config = config
        self.client_manager = None

    def _prepare_payload(self, event: Event, system_info: Dict[str, str]) -> Dict:
        """Prepare the payload for sending."""
        return {
            "event": event.to_dict(),
            "system_info": system_info,
            "timestamp": event.opened,
            "sdk_version": self.config.sdk_version,
        }

    def _should_retry(self, attempt: int, status_code: int = None) -> bool:
        """Determine if a retry should be attempted."""
        if attempt >= self.config.max_retries:
            return False
        if status_code and status_code < 500:  # Don't retry 4xx errors
            return False
        return True

    def _handle_success(self) -> None:
        """Handle successful event sending."""
        pass

    def _handle_failure(
        self, event: Event, system_info: Dict[str, str], exception: Exception
    ) -> None:
        """Handle failed event sending."""
        pass


class BaseAsyncLogClient:
    """Base class for asynchronous logging clients"""

    def __init__(self, config: TracingConfig):
        self.config = config
        self.client_manager = None

    def _prepare_payload(self, event: Event, system_info: Dict[str, str]) -> Dict:
        """Prepare the payload for sending."""
        return {
            "event": event.to_dict(),
            "system_info": system_info,
            "timestamp": event.opened,
            "sdk_version": self.config.sdk_version,
        }

    def _should_retry(self, attempt: int, status_code: int = None) -> bool:
        """Determine if a retry should be attempted."""
        if attempt >= self.config.max_retries:
            return False
        if status_code and status_code < 500:  # Don't retry 4xx errors
            return False
        return True

    def _handle_success(self) -> None:
        """Handle successful event sending."""
        pass

    def _handle_failure(
        self, event: Event, system_info: Dict[str, str], exception: Exception
    ) -> None:
        """Handle failed event sending."""
        pass
