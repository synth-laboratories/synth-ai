"""
Persistent caching for LM responses.

This module provides a SQLite-based cache that persists LM responses
across application restarts, useful for long-term caching of expensive API calls.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, Type, Union

from pydantic import BaseModel

from synth_ai.lm.vendors.base import BaseLMResponse


@dataclass
class PersistentCache:
    """
    Persistent cache implementation using SQLite.
    
    This cache stores LM responses in a SQLite database that persists
    across application restarts.
    """
    def __init__(self, db_path: str = ".cache/persistent_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS cache
                              (key TEXT PRIMARY KEY, response TEXT)""")
        self.conn.commit()

    def hit_cache(
        self, key: str, response_model: Optional[Type[BaseModel]] = None
    ) -> Optional[BaseLMResponse]:
        """
        Check if a response exists in cache for the given key.
        
        Args:
            key: Cache key to look up
            response_model: Optional Pydantic model class to reconstruct structured output
            
        Returns:
            BaseLMResponse if found in cache, None otherwise
        """
        self.cursor.execute("SELECT response FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        if not result:
            return None

        try:
            cache_data = json.loads(result[0])
        except json.JSONDecodeError:
            # Handle legacy string responses
            return BaseLMResponse(raw_response=result[0], structured_output=None, tool_calls=None)

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
        
        Note:
            Uses INSERT OR REPLACE to update existing cache entries.
        """
        if isinstance(response, str):
            cache_data = response
        elif isinstance(response, BaseLMResponse):
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
        else:
            raise ValueError(f"Invalid response type: {type(response)}")

        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
            (key, json.dumps(cache_data)),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection and free resources."""
        self.conn.close()
