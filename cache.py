"""Byte-budgeted LRU cache for rendered PNG bytes.

Keyed by hash(bbox, snapped_time, channel, enhancement).
Evicts oldest entries when either count or byte budget is exceeded.

Latest-time entries get a 5 min TTL — caller passes ``ttl_seconds`` on put().
Other entries are immortal (the underlying GOES file is permanent).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional


class RenderCache:
    def __init__(self, max_entries: int = 200, max_bytes: int = 100 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._store: "OrderedDict[str, tuple[bytes, float]]" = OrderedDict()
        # value: (bytes, expires_at_unix_or_inf)
        self._bytes = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def get(self, key: str) -> Optional[bytes]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            data, expires_at = entry
            if expires_at != float("inf") and expires_at < now:
                # expired
                self._store.pop(key, None)
                self._bytes -= len(data)
                return None
            self._store.move_to_end(key)
            return data

    def put(self, key: str, data: bytes, ttl_seconds: Optional[float] = None) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds else float("inf")
        with self._lock:
            existing = self._store.pop(key, None)
            if existing is not None:
                self._bytes -= len(existing[0])
            self._store[key] = (data, expires_at)
            self._bytes += len(data)
            self._evict_locked()

    def _evict_locked(self) -> None:
        # Evict by count first
        while len(self._store) > self.max_entries:
            _, (data, _) = self._store.popitem(last=False)
            self._bytes -= len(data)
        # Then by byte budget
        while self._bytes > self.max_bytes and self._store:
            _, (data, _) = self._store.popitem(last=False)
            self._bytes -= len(data)
