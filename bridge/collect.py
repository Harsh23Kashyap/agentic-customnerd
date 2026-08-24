"""Stack Overflow collect wrapper with rate-limit / curl resilience."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, List

from bridge.legacy import ensure_legacy_backend

_SEARCH_CACHE: dict[str, list] = {}


def _dedupe_enabled() -> bool:
    raw = os.getenv("AGENT_SEARCH_DEDUPE", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _cache_key(query_list: List[str], max_date: str | None, kwargs: dict) -> str:
    payload = json.dumps(
        {
            "queries": sorted(str(q) for q in (query_list or [])),
            "max_date": max_date,
            "tagged": kwargs.get("tagged"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def collect_with_fallback(
    query_list: List[str],
    *,
    max_date: str | None = None,
    **kwargs: Any,
) -> list:
    """Call legacy collect_articles via collect_articles_with_curl_fallback."""
    ensure_legacy_backend()
    from helper_functions import collect_articles_with_curl_fallback
    from user_search_apis import collect_articles

    if _dedupe_enabled():
        key = _cache_key(query_list, max_date, kwargs)
        cached = _SEARCH_CACHE.get(key)
        if cached is not None:
            print(f"[collect] cache hit key={key} articles={len(cached)}")
            return list(cached)

    result = collect_articles_with_curl_fallback(
        collect_articles,
        query_list,
        max_date=max_date,
        **kwargs,
    ) or []

    if _dedupe_enabled():
        _SEARCH_CACHE[_cache_key(query_list, max_date, kwargs)] = list(result)

    return result


def clear_search_cache() -> None:
    _SEARCH_CACHE.clear()
