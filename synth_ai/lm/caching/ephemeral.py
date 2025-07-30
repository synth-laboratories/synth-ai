"""
Ephemeral (temporary) caching for LM responses.

This module provides a disk-based cache that persists only for the duration
of the application run, useful for avoiding redundant API calls within a session.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union

from diskcache import Cache
from pydantic import BaseModel

from synth_ai.lm.caching.constants import DISKCACHE_SIZE_LIMIT
from synth_ai.lm.vendors.base import BaseLMResponse


@dataclass
class EphemeralCache:
    """
    Ephemeral cache implementation using diskcache.
    
    This cache stores LM responses temporarily on disk with a size limit.
    The cache is cleared when the application restarts.
    """
    def __init__(self, fast_cache_dir: str = ".cache/ephemeral_cache"):
        os.makedirs(fast_cache_dir, exist_ok=True)
        self.fast_cache = Cache(fast_cache_dir, size_limit=DISKCACHE_SIZE_LIMIT)

    def hit_cache(
        self, key: str, response_model: Optional[BaseModel] = None
    ) -> Optional[BaseLMResponse]:
        """
        Check if a response exists in cache for the given key.
        
        Args:
            key: Cache key to look up
            response_model: Optional Pydantic model to reconstruct structured output
            
        Returns:
            BaseLMResponse if found in cache, None otherwise
        """
        if key not in self.fast_cache:
            return None

        try:
            cache_data = self.fast_cache[key]
        except AttributeError:
            return None

        if not isinstance(cache_data, dict):
            return BaseLMResponse(raw_response=cache_data, structured_output=None, tool_calls=None)

        raw_response = cache_data.get("raw_response")
        tool_calls = cache_data.get("tool_calls")
        structured_output = cache_data.get("structured_output")

        if response_model and structured_output:
            structured_output = response_model(**structured_output)

        return BaseLMResponse(
            raw_response=raw_response,
            structured_output=structured_output,
            tool_calls=tool_calls,
        )

    def add_to_cache(self, key: str, response: Union[BaseLMResponse, str]) -> None:
        """
        Add a response to the cache.
        
        Args:
            key: Cache key to store under
            response: Either a BaseLMResponse object or raw string response
            
        Raises:
            ValueError: If response type is not supported
        """
        if isinstance(response, str):
            self.fast_cache[key] = response
            return

        if isinstance(response, BaseLMResponse):
            cache_data = {
                "raw_response": response.raw_response
                if response.raw_response is not None
                else None,
                "tool_calls": response.tool_calls if response.tool_calls is not None else None,
                "structured_output": (
                    response.structured_output.model_dump()
                    if response.structured_output is not None
                    else None
                ),
            }
            self.fast_cache[key] = cache_data
            return

        raise ValueError(f"Invalid response type: {type(response)}")

    def close(self):
        """Close the cache and free resources."""
        self.fast_cache.close()
