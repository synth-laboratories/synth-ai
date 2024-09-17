from zyk.src.lms.caching.handler import CacheHandler

cache_handler = CacheHandler(use_ephemeral_store=True, use_persistent_store=True)
ephemeral_cache_handler = CacheHandler(use_ephemeral_store=True, use_persistent_store=False)

def get_cache_handler(use_ephemeral_cache_only: bool = False):
    if use_ephemeral_cache_only:
        return ephemeral_cache_handler
    else:
        return cache_handler
