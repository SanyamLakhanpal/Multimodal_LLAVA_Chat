from functools import lru_cache

# Adjust the parameters as needed
LRU_CACHE_SIZE = 128
LRU_CACHE_TTL = None  # Time to live (in seconds) for cache entries, None for infinite

def cached(maxsize=LRU_CACHE_SIZE, ttl=LRU_CACHE_TTL):
    """LRU cache decorator with support for distributed environments."""
    def decorator(func):
        return lru_cache(maxsize=maxsize, typed=True, ttl=ttl)(func)
    return decorator
