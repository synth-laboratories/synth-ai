import hashlib
from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel

from zyk.src.lms.caching.ephemeral import EphemeralCache
from zyk.src.lms.caching.persistent import PersistentCache

persistent_cache = PersistentCache()
ephemeral_cache = EphemeralCache()

def map_params_to_key(messages: List[Dict],
        model: str,
        temperature: float,
        response_model: Type[BaseModel]) -> str:
    if not all([isinstance(msg["content"], str) for msg in messages]):
        normalized_messages = "".join([str(msg["content"]) for msg in messages])
    else:
        normalized_messages = "".join([msg["content"] for msg in messages])
    normalized_model = model
    normalized_temperature = f"{temperature:.2f}"[:4]
    normalized_response_model = (
        str(response_model.schema()) if response_model else ""
    )
    return hashlib.sha256(
        (
            normalized_messages
            + normalized_model
            + normalized_temperature
            + normalized_response_model
        ).encode()
    ).hexdigest()

class CacheHandler:
    use_persistent_store: bool = False
    use_ephemeral_store: bool = True

    def __init__(self, use_persistent_store: bool = False, use_ephemeral_store: bool = True):
        self.use_persistent_store = use_persistent_store
        self.use_ephemeral_store = use_ephemeral_store

    def hit_managed_cache(self, model: str, messages: List[Dict[str, Any]], lm_config: Dict[str, Any] = {}) -> str:
        assert isinstance(lm_config, dict), "lm_config must be a dictionary"
        key = map_params_to_key(messages, model, lm_config.get("temperature",0.0), lm_config.get("response_model",None))

        #print(lm_config.get("response_model",None).schema() if lm_config.get("response_model",None) else "None")
        if self.use_persistent_store:
            return persistent_cache.hit_cache(key=key, response_model=lm_config.get("response_model",None))
        elif self.use_ephemeral_store:
            return ephemeral_cache.hit_cache(key=key, response_model=lm_config.get("response_model",None))
        else:
            return None

    def add_to_managed_cache(self, model: str, messages: List[Dict[str, Any]], lm_config: Dict[str, Any], output: Union[str, BaseModel]) -> None:
        assert isinstance(lm_config, dict), "lm_config must be a dictionary"
        key = map_params_to_key(messages, model, lm_config.get("temperature",0.0), lm_config.get("response_model",None))
        if self.use_persistent_store:
            persistent_cache.add_to_cache(key, output)
        if self.use_ephemeral_store:
            ephemeral_cache.add_to_cache(key, output)

