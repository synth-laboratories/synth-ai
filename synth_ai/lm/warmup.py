"""
Model warmup utilities for Synth backend.
Handles model preloading and warmup polling.
"""

import httpx
import asyncio
import logging
import sys
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .config import SynthConfig

logger = logging.getLogger(__name__)


class WarmupStatus:
    """Track warmup status for models with TTL."""

    def __init__(self):
        self._warmed_models: Dict[str, datetime] = {}
        self._ttl = timedelta(minutes=10)  # Consider models warm for 10 minutes

    def is_warm(self, model_name: str) -> bool:
        """Check if model is already warmed up."""
        if model_name in self._warmed_models:
            if datetime.now() - self._warmed_models[model_name] < self._ttl:
                return True
            else:
                # Clean up expired entry
                del self._warmed_models[model_name]
        return False

    def mark_warm(self, model_name: str):
        """Mark model as warmed up."""
        self._warmed_models[model_name] = datetime.now()

    def clear(self):
        """Clear all warmup status."""
        self._warmed_models.clear()


# Global warmup tracker
_warmup_status = WarmupStatus()


async def warmup_synth_model(
    model_name: str,
    config: Optional[SynthConfig] = None,
    max_attempts: Optional[int] = None,
    force: bool = False,
    verbose: bool = True,
    gpu_preference: Optional[str] = None,
) -> bool:
    """
    Warm up a model on the Synth backend using fire-and-forget approach.

    Args:
        model_name: Name of the model to warm up
        config: Synth configuration (loads from env if not provided)
        max_attempts: Maximum number of polling attempts
        force: Force warmup even if model is cached as warm
        verbose: Print progress messages

    Returns:
        True if model is successfully warmed up, False otherwise
    """
    # Load config from environment if not provided
    if config is None:
        config = SynthConfig.from_env()

    # Check if already warm
    if not force and _warmup_status.is_warm(model_name):
        return True

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {config.api_key}"}
        if gpu_preference:
            headers["X-GPU-Preference"] = gpu_preference

        # Step 1: Start warmup (fire and forget)
        try:
            response = await client.post(
                f"{config.get_base_url_without_v1()}/warmup/{model_name}",
                headers=headers,
                timeout=30.0,  # Short timeout for the start request
            )

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") in ["warming", "already_warming"]:
                    pass
                elif response_data.get("status") == "already_warmed":
                    _warmup_status.mark_warm(model_name)
                    return True
                else:
                    logger.warning(f"Unexpected warmup response: {response_data}")
            else:
                logger.warning(
                    f"Warmup start failed with status {response.status_code}: {response.text}"
                )
                return False

        except Exception as e:
            logger.warning(f"Warmup start failed: {e}")
            return False

        # Step 2: Poll status until ready (indefinite by default)
        spinner = "|/-\\"
        spin_idx = 0
        start_time = time.time()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await client.get(
                    f"{config.get_base_url_without_v1()}/warmup/status/{model_name}",
                    headers=headers,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get("status")

                    if status == "warmed":
                        _warmup_status.mark_warm(model_name)
                        # Final spinner line as success
                        elapsed = int(time.time() - start_time)
                        sys.stdout.write(f"\r✅ Warmed {model_name} in {elapsed}s        \n")
                        sys.stdout.flush()
                        return True
                    elif status == "failed":
                        error = status_data.get("error", "Unknown error")
                        logger.error(f"❌ Warmup failed for {model_name}: {error}")
                        sys.stdout.write(f"\r❌ Warmup failed: {error}            \n")
                        sys.stdout.flush()
                        return False
                    else:
                        # Treat unknown statuses (e.g., "cold") as still warming
                        elapsed = int(time.time() - start_time)
                        wheel = spinner[spin_idx % len(spinner)]
                        spin_idx += 1
                        label = status or "pending"
                        sys.stdout.write(
                            f"\r⏳ Warming {model_name} [{wheel}] status={label} elapsed={elapsed}s"
                        )
                        sys.stdout.flush()

                # Short sleep between status checks
                await asyncio.sleep(2.0)

            except httpx.TimeoutException:
                # Continue polling; update spinner line
                elapsed = int(time.time() - start_time)
                wheel = spinner[spin_idx % len(spinner)]
                spin_idx += 1
                sys.stdout.write(
                    f"\r⏳ Warming {model_name} [{wheel}] status=timeout elapsed={elapsed}s"
                )
                sys.stdout.flush()
                await asyncio.sleep(1.0)
            except Exception as e:
                # Continue polling; update spinner line with error label
                elapsed = int(time.time() - start_time)
                wheel = spinner[spin_idx % len(spinner)]
                spin_idx += 1
                sys.stdout.write(
                    f"\r⏳ Warming {model_name} [{wheel}] status=error elapsed={elapsed}s"
                )
                sys.stdout.flush()
                await asyncio.sleep(1.0)

            # Optional max_attempts for callers who want a cap
            if max_attempts is not None and attempt >= max_attempts:
                logger.error(f"Failed to warm up {model_name} after {max_attempts} status checks")
                sys.stdout.write("\n")
                sys.stdout.flush()
                return False


def get_warmup_status() -> WarmupStatus:
    """Get the global warmup status tracker."""
    return _warmup_status
