from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BaseLMResponse(BaseModel):
    raw_response: str
    structured_output: Optional[BaseModel] = None
    tool_calls: Optional[List[Dict]] = None


class VendorBase(ABC):
    used_for_structured_outputs: bool = False
    exceptions_to_retry: List[Exception] = []

    @abstractmethod
    async def _hit_api_async(
        self,
        messages: List[Dict[str, Any]],
        response_model_override: Optional[BaseModel] = None,
    ) -> str:
        pass

    @abstractmethod
    def _hit_api_sync(
        self,
        messages: List[Dict[str, Any]],
        response_model_override: Optional[BaseModel] = None,
    ) -> str:
        pass
