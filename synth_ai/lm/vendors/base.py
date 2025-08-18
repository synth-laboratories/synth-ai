"""
Base classes for LM vendors.

This module provides abstract base classes for implementing language model vendor integrations.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseLMResponse(BaseModel):
    """
    Standard response format from language model API calls.

    Attributes:
        raw_response: The raw text response from the model
        structured_output: Optional parsed Pydantic model if structured output was requested
        tool_calls: Optional list of tool calls if tools were provided
        response_id: Optional response ID for thread management (Responses API)
        reasoning: Optional reasoning trace from the model (o1 models)
        api_type: Optional API type used ("chat", "responses", or "harmony")
    """

    raw_response: str
    structured_output: BaseModel | None = None
    tool_calls: list[dict] | None = None
    response_id: str | None = None
    reasoning: str | None = None
    api_type: str | None = None
    usage: dict[str, Any] | None = None


class VendorBase(ABC):
    """
    Abstract base class for language model vendor implementations.

    Attributes:
        used_for_structured_outputs: Whether this vendor supports structured outputs
        exceptions_to_retry: List of exceptions that should trigger retries
    """

    used_for_structured_outputs: bool = False
    exceptions_to_retry: list[Exception] = []

    @abstractmethod
    async def _hit_api_async(
        self,
        messages: list[dict[str, Any]],
        response_model_override: BaseModel | None = None,
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
        messages: list[dict[str, Any]],
        response_model_override: BaseModel | None = None,
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
