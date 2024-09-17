import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

from diskcache import Cache
from pydantic import BaseModel

from zyk.src.lms.caching.constants import DISKCACHE_SIZE_LIMIT


@dataclass
class EphemeralCache:
    def __init__(self, fast_cache_dir: str = ".cache/ephemeral_cache"):
        os.makedirs(fast_cache_dir, exist_ok=True)
        self.fast_cache = Cache(fast_cache_dir, size_limit=DISKCACHE_SIZE_LIMIT)

    def hit_cache(self, key: str, response_model: BaseModel) -> Optional[Dict]:
        assert isinstance(response_model, BaseModel)
        if key in self.fast_cache:
            try:
                cache_data = self.fast_cache[key]
            except AttributeError:
                return None
            if response_model is not None:
                if isinstance(cache_data["response"], dict):
                    response = cache_data["response"]
                    return response_model(**response)
            if isinstance(cache_data, str):
                print("Cache hit: ", cache_data)
                return cache_data
            #     # cache_data = {
            #         "response": cache_data,
            #         "response_class": None,
            #     }
            # return cache_data["response"]
        return None

    def add_to_cache(self, key: str, response: Union[BaseModel, str]) -> None:

        if isinstance(response, BaseModel):
            response_dict = response.model_dump()
            response_class = response.__class__.__name__
        elif isinstance(response, str):
            response_dict = response
            response_class = None
        else:
            raise ValueError(f"Invalid response type: {type(response)}")

        cache_data = {
            "response": response_dict,
            "response_class": response_class,
        }
        self.fast_cache[key] = cache_data
        return key

    def close(self):
        self.fast_cache.close()