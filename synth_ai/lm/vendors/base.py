"""
Base classes for LM vendors.

This module provides abstract base classes for implementing language model vendor integrations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BaseLMResponse(BaseModel):
    """
    Standard response format from language model API calls.
    
    Attributes:
        raw_response: The raw text response from the model
        structured_output: Optional parsed Pydantic model if structured output was requested
        tool_calls: Optional list of tool calls if tools were provided
    """
    raw_response: str
    structured_output: Optional[BaseModel] = None
    tool_calls: Optional[List[Dict]] = None


class VendorBase(ABC):
    """
    Abstract base class for language model vendor implementations.
    
    Attributes:
        used_for_structured_outputs: Whether this vendor supports structured outputs
        exceptions_to_retry: List of exceptions that should trigger retries
    """
    used_for_structured_outputs: bool = False
    exceptions_to_retry: List[Exception] = []

    @abstractmethod
    async def _hit_api_async(
        self,
        messages: List[Dict[str, Any]],
        response_model_override: Optional[BaseModel] = None,
    ) -> str:
        """
        Make an asynchronous API call to the language model.
        
        Args:
            messages: List of message dictionaries with role and content
            response_model_override: Optional Pydantic model for structured output
            
        Returns:
            str: The model's response
        """
        pass

    @abstractmethod
    def _hit_api_sync(
        self,
        messages: List[Dict[str, Any]],
        response_model_override: Optional[BaseModel] = None,
    ) -> str:
        """
        Make a synchronous API call to the language model.
        
        Args:
            messages: List of message dictionaries with role and content
            response_model_override: Optional Pydantic model for structured output
            
        Returns:
            str: The model's response
        """
        pass
