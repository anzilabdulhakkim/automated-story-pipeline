"""
cache.py — Two-layer response cache.

Layer 1 — Exact-match (primary)
    Key   : SHA-256(model + system_prompt + user_prompt)
    Store : diskcache (SQLite-backed, survives restarts)
    TTL   : configurable, default 1 hour
    Use   : identical story configs → instant reply, zero API cost

Layer 2 — Session-scoped in-memory index
    A lightweight dict kept in process memory so the batch runner can
    detect repeated configs within a run without a disk round-trip.

Usage
-----
    from cache import response_cache

    hit = response_cache.get(model_name, system_prompt, user_prompt)
    if hit:
        return hit  # no API call needed

    ... call API ...

    response_cache.set(model_name, system_prompt, user_prompt, result)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional, TYPE_CHECKING

# diskcache is the preferred backend; fall back to a plain JSON file when not
# installed (e.g. first run before pip install).
try:
    import diskcache as _diskcache
    _DISKCACHE_OK = True
except ImportError:
    _diskcache = None
    _DISKCACHE_OK = False

if TYPE_CHECKING:
    import diskcache as diskcache_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_extra(extra: Any) -> str:
    if extra is None:
        return ""
    if isinstance(extra, str):
        return extra
    try:
        return json.dumps(extra, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(extra)


def _make_key(model: str, system_prompt: str, user_prompt: str, extra: Any = None) -> str:
    """Deterministic SHA-256 key from model + prompt + request fingerprint."""
    raw = f"{model}\x00{system_prompt or ''}\x00{user_prompt}\x00{_normalize_extra(extra)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main Cache Class
# ---------------------------------------------------------------------------

class PipelineCache:
    """
    Persistent exact-match cache for Gemini API responses.

    Parameters
    ----------
    cache_dir
        Directory where cache data is stored.
    ttl_seconds
        Time-to-live for cached entries (seconds).  Tie this to your data
        update frequency, not a fixed timer — see guide §3.
    """

    def __init__(self, cache_dir: str = ".cache", ttl_seconds: int = 3_600) -> None:
        self.cache_dir   = cache_dir
        self.ttl_seconds = ttl_seconds

        # In-process session index (key → value) for instant repeated-config detection;
        # stores actual values so in-memory hits skip the disk read entirely.
        self._session_index: dict[str, Any] = {}

        os.makedirs(cache_dir, exist_ok=True)

        if _DISKCACHE_OK and _diskcache is not None:
            self._disk: "diskcache_module.Cache" = _diskcache.Cache(cache_dir)
        else:
            # Fallback: plain JSON file
            self._json_path = os.path.join(cache_dir, "response_cache.json")
            self._mem: dict[str, dict] = self._load_json()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        extra: Any = None,
    ) -> Optional[Any]:
        """Return cached response or ``None`` on miss."""
        key = _make_key(model, system_prompt, user_prompt, extra)

        # Fast in-process check: return the stored value without disk I/O
        if key in self._session_index:
            return self._session_index[key]

        return self._fetch(key)

    def set(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        value: Any,
        extra: Any = None,
    ) -> None:
        """Store a response in the cache."""
        key = _make_key(model, system_prompt, user_prompt, extra)
        self._session_index[key] = value  # store value, not just True
        self._store(key, value)

    def invalidate(self, model: str, system_prompt: str, user_prompt: str, extra: Any = None) -> None:
        """Explicitly evict a cached entry (e.g. after prompt change)."""
        key = _make_key(model, system_prompt, user_prompt, extra)
        self._session_index.pop(key, None)
        if _DISKCACHE_OK:
            self._disk.delete(key)
        else:
            self._mem.pop(key, None)
            self._save_json()

    def clear(self) -> None:
        """Wipe the entire cache (useful in tests)."""
        self._session_index.clear()
        if _DISKCACHE_OK:
            self._disk.clear()
        else:
            self._mem.clear()
            self._save_json()

    @property
    def backend(self) -> str:
        return "diskcache" if _DISKCACHE_OK else "json-file"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch(self, key: str) -> Optional[Any]:
        if _DISKCACHE_OK:
            return self._disk.get(key)  # returns None on miss or expired

        entry = self._mem.get(key)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            del self._mem[key]
            self._save_json()
            return None
        return entry["value"]

    def _store(self, key: str, value: Any) -> None:
        if _DISKCACHE_OK:
            self._disk.set(key, value, expire=self.ttl_seconds)
        else:
            self._mem[key] = {
                "value":      value,
                "expires_at": time.time() + self.ttl_seconds,
            }
            self._save_json()

    def _load_json(self) -> dict:
        if os.path.exists(self._json_path):
            try:
                with open(self._json_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_json(self) -> None:
        try:
            with open(self._json_path, "w", encoding="utf-8") as fh:
                json.dump(self._mem, fh, indent=2, ensure_ascii=False)
        except OSError:
            pass  # non-fatal — cache is best-effort


# Module-level singleton
response_cache = PipelineCache()
