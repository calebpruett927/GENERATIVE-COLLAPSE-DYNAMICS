"""Content-addressable, multi-node artifact cache.

Design
~~~~~~
* **Content-addressable**: every artifact is stored under its SHA256 digest.
  Identical validation outputs on any node hash to the same key.
* **Pluggable backend**: in-memory dict (default), filesystem (disk), or
  Redis (optional dependency).  All backends implement ``CacheBackend``.
* **Tenant isolation**: cache keys are scoped to a tenant namespace.
  Tenant A cannot read Tenant B's artifacts even if the digests collide.
* **Eviction**: LRU eviction when the cache exceeds ``max_bytes``.
* **TTL**: entries older than ``ttl_s`` are lazily evicted on access.

Thread safety: all operations are protected by a ``threading.Lock``.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from umcp.fleet.models import CacheEntry

# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for pluggable cache storage backends."""

    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, data: bytes) -> None: ...
    def delete(self, key: str) -> bool: ...
    def keys(self) -> list[str]: ...
    def size(self, key: str) -> int: ...
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend (default)
# ---------------------------------------------------------------------------


class MemoryBackend:
    """Simple in-memory dict backend — single process only."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def put(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def keys(self) -> list[str]:
        return list(self._store)

    def size(self, key: str) -> int:
        d = self._store.get(key)
        return len(d) if d is not None else 0

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Filesystem backend
# ---------------------------------------------------------------------------


class FilesystemBackend:
    """Filesystem-backed cache — survives process restarts.

    Layout::

        root/
          <key[:2]>/
            <key>.bin
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.bin"

    def get(self, key: str) -> bytes | None:
        p = self._path(key)
        if p.exists():
            return p.read_bytes()
        return None

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if p.exists():
            p.unlink()
            return True
        return False

    def keys(self) -> list[str]:
        result: list[str] = []
        if self._root.exists():
            for f in self._root.rglob("*.bin"):
                result.append(f.stem)
        return result

    def size(self, key: str) -> int:
        p = self._path(key)
        return p.stat().st_size if p.exists() else 0

    def clear(self) -> None:
        if self._root.exists():
            shutil.rmtree(self._root)
            self._root.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Redis backend (optional)
# ---------------------------------------------------------------------------


class RedisBackend:
    """Redis-backed cache — multi-node shared storage.

    Requires the ``redis`` package::

        pip install redis

    Parameters
    ----------
    url
        Redis connection URL (e.g. ``redis://localhost:6379/0``).
    prefix
        Key prefix for namespacing in a shared Redis instance.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "umcp:cache:") -> None:
        try:
            import redis as _redis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Redis backend requires the 'redis' package: pip install redis") from exc

        self._client = _redis.from_url(url)
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> bytes | None:
        val = self._client.get(self._key(key))
        return val if isinstance(val, bytes) else None

    def put(self, key: str, data: bytes) -> None:
        self._client.set(self._key(key), data)

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(self._key(key)))

    def keys(self) -> list[str]:
        prefix_len = len(self._prefix)
        raw_keys = self._client.keys(f"{self._prefix}*")
        return [k.decode()[prefix_len:] if isinstance(k, bytes) else k[prefix_len:] for k in raw_keys]

    def size(self, key: str) -> int:
        length = self._client.strlen(self._key(key))
        return int(length) if length else 0

    def clear(self) -> None:
        raw_keys = self._client.keys(f"{self._prefix}*")
        if raw_keys:
            self._client.delete(*raw_keys)


# ---------------------------------------------------------------------------
# ArtifactCache — public API
# ---------------------------------------------------------------------------


@dataclass
class _CacheMeta:
    """Mutable metadata for an in-flight cache entry."""

    entry: CacheEntry
    last_accessed: float = 0.0
    hit_count: int = 0


class ArtifactCache:
    """Content-addressable, tenant-scoped artifact cache with LRU eviction.

    Parameters
    ----------
    backend
        Storage backend (default: in-memory).
    max_bytes
        Maximum total cache size.  0 = unbounded.
    ttl_s
        Time-to-live in seconds.  0 = no expiry.
    """

    def __init__(
        self,
        *,
        backend: CacheBackend | None = None,
        max_bytes: int = 0,
        ttl_s: float = 0,
    ) -> None:
        self._backend: CacheBackend = backend or MemoryBackend()
        self._max_bytes = max_bytes
        self._ttl_s = ttl_s
        self._meta: dict[str, _CacheMeta] = {}
        self._total_bytes = 0
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def content_key(data: bytes) -> str:
        """SHA256-based content-addressable key."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _scoped_key(tenant_id: str, key: str) -> str:
        """Scope a key to a tenant namespace."""
        if tenant_id:
            return f"{tenant_id}/{key}"
        return key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, *, tenant_id: str = "") -> bytes | None:
        """Retrieve an artifact by key, scoped to tenant."""
        scoped = self._scoped_key(tenant_id, key)
        with self._lock:
            meta = self._meta.get(scoped)
            if meta is None:
                self._misses += 1
                return None

            # TTL check
            if self._ttl_s and (time.time() - meta.entry.created_at) > self._ttl_s:
                self._evict_locked(scoped)
                self._misses += 1
                return None

            data = self._backend.get(scoped)
            if data is None:
                # Backend lost the entry — clean up metadata
                self._meta.pop(scoped, None)
                self._misses += 1
                return None

            meta.last_accessed = time.time()
            meta.hit_count += 1
            self._hits += 1
            return data

    def put(
        self,
        data: bytes,
        *,
        key: str | None = None,
        tenant_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store an artifact.  Returns the content-addressable key.

        If ``key`` is not given, it is derived from the SHA256 of ``data``.
        """
        if key is None:
            key = self.content_key(data)
        scoped = self._scoped_key(tenant_id, key)

        with self._lock:
            size = len(data)

            # Evict if over budget
            if self._max_bytes:
                while self._total_bytes + size > self._max_bytes and self._meta:
                    self._evict_lru_locked()

            self._backend.put(scoped, data)
            now = time.time()
            entry = CacheEntry(
                key=key,
                tenant_id=tenant_id,
                size_bytes=size,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {},
            )
            self._meta[scoped] = _CacheMeta(entry=entry, last_accessed=now)
            self._total_bytes += size
            return key

    def delete(self, key: str, *, tenant_id: str = "") -> bool:
        """Remove an artifact by key."""
        scoped = self._scoped_key(tenant_id, key)
        with self._lock:
            return self._evict_locked(scoped)

    def has(self, key: str, *, tenant_id: str = "") -> bool:
        """Check if a key exists (without updating access time)."""
        scoped = self._scoped_key(tenant_id, key)
        with self._lock:
            return scoped in self._meta

    def tenant_usage(self, tenant_id: str) -> int:
        """Total bytes used by a tenant."""
        with self._lock:
            return sum(m.entry.size_bytes for m in self._meta.values() if m.entry.tenant_id == tenant_id)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> dict[str, Any]:
        """Return stats — caller must hold ``_lock``."""
        return {
            "total_entries": len(self._meta),
            "total_bytes": self._total_bytes,
            "max_bytes": self._max_bytes,
            "ttl_s": self._ttl_s,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (round(self._hits / (self._hits + self._misses), 4) if (self._hits + self._misses) else 0.0),
        }

    def clear(self, *, tenant_id: str = "") -> int:
        """Clear all entries (or all for a specific tenant).  Returns count."""
        with self._lock:
            if tenant_id:
                to_evict = [k for k, m in self._meta.items() if m.entry.tenant_id == tenant_id]
                for k in to_evict:
                    self._evict_locked(k)
                return len(to_evict)
            else:
                count = len(self._meta)
                self._meta.clear()
                self._total_bytes = 0
                self._backend.clear()
                return count

    def to_dict(self) -> dict[str, Any]:
        """Serialise cache state for diagnostics."""
        with self._lock:
            return {
                "stats": self._stats_unlocked(),
                "entries": [m.entry.to_dict() for m in self._meta.values()],
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_locked(self, scoped_key: str) -> bool:
        """Evict a single entry.  Caller must hold _lock."""
        meta = self._meta.pop(scoped_key, None)
        if meta is None:
            return False
        self._backend.delete(scoped_key)
        self._total_bytes -= meta.entry.size_bytes
        return True

    def _evict_lru_locked(self) -> None:
        """Evict the least-recently-used entry.  Caller must hold _lock."""
        if not self._meta:
            return
        lru_key = min(self._meta, key=lambda k: self._meta[k].last_accessed)
        self._evict_locked(lru_key)
