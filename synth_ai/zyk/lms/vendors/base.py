from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class VendorBase(ABC):
    used_for_structured_outputs: bool = False
    exceptions_to_retry: List[Exception] = []
    
    @abstractmethod
    async def _hit_api_async(self, messages: List[Dict[str, Any]], response_model_override: Optional[BaseModel] = None) -> str:
        pass
    
    @abstractmethod
    def _hit_api_sync(self, messages: List[Dict[str, Any]], response_model_override: Optional[BaseModel] = None) -> str:
        pass