import hashlib
import json
import sys
import types
import bridge.collect as collect


def test_cache_key_is_order_independent():
    assert collect._cache_key(["b", "a"], None, {}) == collect._cache_key(["a", "b"], None, {})


def test_dedupe_env(monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_DEDUPE", "off")
    assert collect._dedupe_enabled() is False


def test_collect_caches_without_second_call(monkeypatch):
    calls = []
    helper = types.ModuleType("helper_functions")
    helper.collect_articles_with_curl_fallback = lambda fn, qs, max_date=None, **kw: calls.append(list(qs)) or [{"id": 1}]
    user = types.ModuleType("user_search_apis")
    user.collect_articles = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "helper_functions", helper)
    monkeypatch.setitem(sys.modules, "user_search_apis", user)
    monkeypatch.setattr(collect, "ensure_legacy_backend", lambda: None)
    monkeypatch.setenv("AGENT_SEARCH_DEDUPE", "true")
    collect.clear_search_cache()
    one = collect.collect_with_fallback(["q"])
    two = collect.collect_with_fallback(["q"])
    assert one == two == [{"id": 1}]
    assert calls == [["q"]]
    assert one is not two
