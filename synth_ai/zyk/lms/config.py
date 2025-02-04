import os
from dotenv import load_dotenv


def should_use_cache() -> bool:
    load_dotenv()
    cache_env = os.getenv("USE_ZYK_CACHE", "true").lower()
    return cache_env not in ("false", "0", "no")
