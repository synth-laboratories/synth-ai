"""Utility functions for storage layer."""

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

T = TypeVar("T")


def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying async functions on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Backoff multiplier for each retry
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic failed without exception")

        return wrapper

    return decorator


async def batch_process(
    items: list[Any], processor: Callable, batch_size: int = 100, max_concurrent: int = 5
) -> list[Any]:
    """Process items in batches with concurrency control.

    Args:
        items: List of items to process
        processor: Async function to process a batch
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent batches

    Returns:
        List of results from all batches
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def process_batch(batch):
        async with semaphore:
            return await processor(batch)

    # Create batches
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    # Process all batches
    batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])

    # Flatten results
    for result in batch_results:
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

    return results


def sanitize_json(data: Any) -> Any:
    """Sanitize data for JSON storage.

    Converts non-JSON-serializable types to strings.
    """
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(item) for item in data]
    elif isinstance(data, str | int | float | bool | type(None)):
        return data
    else:
        return str(data)


def estimate_size(obj: Any) -> int:
    """Estimate the size of an object in bytes.

    This is a rough estimate for storage planning.
    """
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    elif isinstance(obj, int | float):
        return 8
    elif isinstance(obj, bool):
        return 1
    elif isinstance(obj, dict):
        size = 0
        for k, v in obj.items():
            size += estimate_size(k) + estimate_size(v)
        return size
    elif isinstance(obj, list):
        return sum(estimate_size(item) for item in obj)
    else:
        return len(str(obj).encode("utf-8"))


class StorageMetrics:
    """Track storage operation metrics."""

    def __init__(self):
        self.operations: dict[str, dict[str, Any]] = {}

    def record_operation(
        self, operation: str, duration: float, success: bool, size: int | None = None
    ):
        """Record a storage operation."""
        if operation not in self.operations:
            self.operations[operation] = {
                "count": 0,
                "success_count": 0,
                "total_duration": 0.0,
                "total_size": 0,
            }

        stats = self.operations[operation]
        stats["count"] += 1
        if success:
            stats["success_count"] += 1
        stats["total_duration"] += duration
        if size:
            stats["total_size"] += size

    def get_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get statistics for operations."""
        if operation:
            stats = self.operations.get(operation, {})
            if stats:
                return {
                    "count": stats["count"],
                    "success_rate": stats["success_count"] / stats["count"]
                    if stats["count"] > 0
                    else 0,
                    "avg_duration": stats["total_duration"] / stats["count"]
                    if stats["count"] > 0
                    else 0,
                    "total_size": stats["total_size"],
                }
            return {}
        else:
            return {op: self.get_stats(op) for op in self.operations}


# Global metrics instance
STORAGE_METRICS = StorageMetrics()


def track_metrics(operation: str):
    """Decorator to track storage operation metrics."""

    def decorator(func: Callable[..., Awaitable[T]] | Callable[..., T]) -> Callable[..., Awaitable[T]] | Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            success = False
            try:
                async_func = cast(Callable[..., Awaitable[T]], func)
                result = await async_func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                STORAGE_METRICS.record_operation(operation, duration, success)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            success = False
            try:
                sync_func = cast(Callable[..., T], func)
                result = sync_func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                STORAGE_METRICS.record_operation(operation, duration, success)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
