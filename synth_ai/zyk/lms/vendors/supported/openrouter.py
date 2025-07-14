import os
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI, OpenAI
from synth_ai.zyk.lms.vendors.openai_standard import OpenAIStandard
from synth_ai.zyk.lms.vendors.base import BaseLMResponse
from synth_ai.zyk.lms.tools.base import BaseTool


class OpenRouterAPI(OpenAIStandard):
    """OpenRouter API client for accessing various models through OpenRouter's unified API."""
    
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # OpenRouter requires specific headers
        default_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://github.com/synth-laboratories/synth-ai"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "synth-ai")
        }
        
        super().__init__(
            sync_client=OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers=default_headers
            ),
            async_client=AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers=default_headers
            )
        )
    
    def _strip_prefix(self, model: str) -> str:
        """Remove the 'openrouter/' prefix from model names."""
        if model.startswith("openrouter/"):
            return model[len("openrouter/"):]
        return model
    
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        # Strip the 'openrouter/' prefix before calling the API
        model = self._strip_prefix(model)
        return await super()._hit_api_async(
            model, messages, lm_config, use_ephemeral_cache_only, reasoning_effort, tools
        )
    
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        # Strip the 'openrouter/' prefix before calling the API
        model = self._strip_prefix(model)
        return super()._hit_api_sync(
            model, messages, lm_config, use_ephemeral_cache_only, reasoning_effort, tools
        )