"""
JalVaani API — in-process TTL cache.

Thread-safe, no external dependencies.
For multi-process deployments replace with Redis (redis.asyncio + aioredis).

Usage:
    from cache import cache
    val = cache.get("national_stats")
    if val is None:
        val = compute_expensive_thing()
        cache.set("national_stats", val, ttl_seconds=3600)
"""
import time
from threading import Lock
from typing import Any, Optional


class _TTLCache:
    """Sliding-window TTL cache backed by a plain dict + monotonic clock."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() < expires_at:
                return value
            # Expired — evict
            del self._store[key]
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl_seconds)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton — import this everywhere
cache = _TTLCache()
