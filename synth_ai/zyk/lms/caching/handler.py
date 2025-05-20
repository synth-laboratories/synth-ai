import hashlib
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from synth_ai.zyk.lms.caching.ephemeral import EphemeralCache
from synth_ai.zyk.lms.caching.persistent import PersistentCache
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse

persistent_cache = PersistentCache()
ephemeral_cache = EphemeralCache()

import logging
logger = logging.getLogger(__name__)

def map_params_to_key(
    messages: List[Dict],
    model: str,
    temperature: float,
    response_model: Optional[Type[BaseModel]],
    tools: Optional[List[BaseTool]] = None,
    reasoning_effort: str = "low",
) -> str:
    if any(m is None for m in messages):
        raise ValueError("Messages cannot contain None values - messages: ", messages)
    if not all([isinstance(msg["content"], str) for msg in messages]):
        normalized_messages = "".join([str(msg["content"]) for msg in messages])
    else:
        normalized_messages = "".join([msg["content"] for msg in messages])
    normalized_model = model
    normalized_temperature = f"{temperature:.2f}"[:4]
    normalized_response_model = str(response_model.schema()) if response_model else ""
    normalized_reasoning_effort = reasoning_effort if reasoning_effort else ""

    # Normalize tools if present
    normalized_tools = ""
    if tools and all(isinstance(tool, BaseTool) for tool in tools):
        tool_schemas = []
        for tool in sorted(tools, key=lambda x: x.name):  # Sort by name for consistency
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "arguments": tool.arguments.schema(),
            }
            tool_schemas.append(str(tool_schema))
        #logger.error(f"Tool schemas: {tool_schemas}")
        normalized_tools = "".join(tool_schemas)
    elif tools:
        #logger.error(f"Tools: {tools}")
        normalized_tools = "".join([str(tool) for tool in tools])

    key_str = ""
    components = [
        normalized_messages,
        normalized_model,
        normalized_temperature,
        normalized_response_model,
        normalized_tools,
        normalized_reasoning_effort
    ]
    for component in components:
        if component:
            key_str += str(component)

    return hashlib.sha256(key_str.encode()).hexdigest() 


class CacheHandler:
    use_persistent_store: bool = False
    use_ephemeral_store: bool = True

    def __init__(
        self, use_persistent_store: bool = False, use_ephemeral_store: bool = True
    ):
        self.use_persistent_store = use_persistent_store
        self.use_ephemeral_store = use_ephemeral_store

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Validate that messages are in the correct format."""
        assert all(
            [type(msg["content"]) == str for msg in messages]
        ), "All message contents must be strings"

    def hit_managed_cache(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        tools: Optional[List[BaseTool]] = None,
    ) -> Optional[BaseLMResponse]:
        """Hit the cache with the given key."""
        self._validate_messages(messages)
        assert type(lm_config) == dict, "lm_config must be a dictionary"
        key = map_params_to_key(
            messages,
            model,
            lm_config.get("temperature", 0.0),
            lm_config.get("response_model", None),
            tools,
            lm_config.get("reasoning_effort", "low"),
        )
        if self.use_persistent_store:
            return persistent_cache.hit_cache(
                key=key, response_model=lm_config.get("response_model", None)
            )
        elif self.use_ephemeral_store:
            return ephemeral_cache.hit_cache(
                key=key, response_model=lm_config.get("response_model", None)
            )
        else:
            return None

    def add_to_managed_cache(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        output: BaseLMResponse,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        """Add the given output to the cache."""
        self._validate_messages(messages)
        assert type(output) == BaseLMResponse, "output must be a BaseLMResponse"
        assert type(lm_config) == dict, "lm_config must be a dictionary"
        key = map_params_to_key(
            messages,
            model,
            lm_config.get("temperature", 0.0),
            lm_config.get("response_model", None),
            tools,
            lm_config.get("reasoning_effort", "low"),
        )
        if self.use_persistent_store:
            persistent_cache.add_to_cache(key, output)
        if self.use_ephemeral_store:
            ephemeral_cache.add_to_cache(key, output)
