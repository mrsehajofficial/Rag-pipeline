"""Thread-safe LRU cache shared by the embedding and LLM layers.

Bounded memory is the whole point: an unbounded dict in a long-running service is a
slow OOM. Two implementations -- an OrderedDict one (stdlib) and an O(1) doubly
linked list one used when many threads hammer the same key.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Callable, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, max_size: int = 10_000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return default

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)  # evict LRU

    def get_or_compute(self, key: K, factory: Callable[[], V]) -> V:
        """Compute-once. Holds the lock so concurrent callers don't stampede the API."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            value = factory()
            self._data[key] = value
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)
            return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def stable_hash(*parts: Any, length: int = 32) -> str:
    """Deterministic hash across processes. Python's hash() is salted per-process,
    so it can never be used for a persistent cache key."""
    h = hashlib.blake2b(digest_size=32)
    for part in parts:
        h.update(repr(part).encode("utf-8", errors="replace"))
        h.update(b"\x1f")  # unit separator, prevents ("ab","c") == ("a","bc")
    return h.hexdigest()[:length]
