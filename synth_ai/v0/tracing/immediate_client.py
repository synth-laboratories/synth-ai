import asyncio
import logging
import time

import httpx

from .abstractions import Event
from .client_manager import ClientManager
from .config import TracingConfig
from .log_client_base import BaseAsyncLogClient, BaseLogClient

logger = logging.getLogger(__name__)


class ImmediateLogClient(BaseLogClient):
    """Synchronous client for immediate logging of events"""

    def __init__(self, config: TracingConfig):
        super().__init__(config)
        self.client_manager = ClientManager.initialize(config)

    def send_event(self, event: Event, system_info: dict[str, str]) -> bool:
        """Send a single event with retries and fallback"""
        from .retry_queue import (
            retry_queue,  # Import here to avoid circular import
        )

        payload = self._prepare_payload(event, system_info)
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client_manager.sync_client.post(
                    f"{self.config.base_url}/v1/uploads/stream",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                logger.info(f"Event upload response status: {response.status_code}")
                response.raise_for_status()
                response_data = response.json()
                event.id = response_data.get("event_id")  # Store the event_id
                self._handle_success()
                return True
            except Exception as e:
                last_exception = e
                status_code = getattr(e, "response", None)
                if status_code:
                    status_code = status_code.status_code

                if not self._should_retry(attempt, status_code):
                    break

                backoff = self.client_manager.calculate_backoff(attempt)
                time.sleep(backoff)

        # If we get here, all immediate retries failed
        self._handle_failure(event, system_info, last_exception)

        # Add to retry queue for later processing
        retry_queue.add_failed_event(event, system_info)
        return False


class AsyncImmediateLogClient(BaseAsyncLogClient):
    """Asynchronous client for immediate logging of events"""

    def __init__(self, config: TracingConfig):
        super().__init__(config)
        self.client_manager = ClientManager.initialize(config)

    async def send_event(self, event: Event, system_info: dict[str, str]) -> bool:
        """Send a single event with retries and fallback (async version)"""
        from .retry_queue import retry_queue

        if not self.config.api_key:
            logger.error("No API key provided")
            return False

        # First get JWT token
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                auth_response = await client.get(
                    f"{self.config.base_url}/v1/auth/token",
                    headers={"customer_specific_api_key": self.config.api_key.strip()},
                )
                auth_response.raise_for_status()
                auth_data = auth_response.json()  # This is synchronous in httpx
                logger.debug(f"Auth response data: {auth_data}")

                token = auth_data.get("access_token")
                if not token:
                    logger.error(
                        f"No access token received from auth endpoint. Response data: {auth_data}"
                    )
                    return False
            except Exception:
                # logger.error(f"Failed to get auth token: {e}")
                return False

            # Now send the event with the JWT token
            payload = self._prepare_payload(event, system_info)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            logger.debug(f"Request URL: {self.config.base_url}/v1/uploads/stream")
            logger.debug("Using JWT token for authentication")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Payload size: {len(str(payload))} bytes")

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                for attempt in range(self.config.max_retries + 1):
                    try:
                        response = await client.post(
                            f"{self.config.base_url}/v1/uploads/stream",
                            json=payload,
                            headers=headers,
                            timeout=self.config.timeout,
                        )
                        logger.info(f"Event upload response status: {response.status_code}")

                        if response.status_code >= 500:
                            error_body = response.text
                            logger.error(f"Server error. Response: {error_body}")
                            response.raise_for_status()

                        if response.status_code == 401:
                            error_body = response.text
                            logger.error(f"Authentication failed. Response: {error_body}")
                            response.raise_for_status()

                        response_data = response.json()  # This is synchronous in httpx
                        event.id = response_data.get("event_id")
                        return True

                    except Exception:
                        # last_exception = e
                        # logger.error(f"Upload attempt {attempt + 1} failed: {str(e)}")
                        if attempt < self.config.max_retries:
                            backoff = self.client_manager.calculate_backoff(attempt)
                            await asyncio.sleep(backoff)

                # logger.error(
                #     f"All upload attempts failed. Last error: {last_exception}"
                # )
                retry_queue.add_failed_event(event, system_info)
                return False
