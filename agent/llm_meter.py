"""Thread-safe per-request LLM call meter.

Patches the OpenAI SDK at the class level so every chat-completion and
embedding call (legacy backend + agent tools + orchestrator) is counted,
regardless of which client instance issues it.

The counter is a process-global reset at the start of each request. This is
correct as long as requests are processed sequentially (as the eval harness
does). Concurrent requests would share the counter.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_state = {
    "chat": 0,
    "embeddings": 0,
    "by_model": {},
}
_installed = False


def reset() -> None:
    with _lock:
        _state["chat"] = 0
        _state["embeddings"] = 0
        _state["by_model"] = {}


def incr_chat(model: str = "unknown", n: int = 1) -> None:
    with _lock:
        _state["chat"] += n
        _state["by_model"][model] = _state["by_model"].get(model, 0) + n


def incr_embeddings(model: str = "unknown", n: int = 1) -> None:
    with _lock:
        _state["embeddings"] += n


def snapshot() -> dict:
    with _lock:
        return {
            "chat": _state["chat"],
            "embeddings": _state["embeddings"],
            "total": _state["chat"] + _state["embeddings"],
            "by_model": dict(_state["by_model"]),
        }


def install() -> None:
    """Monkeypatch OpenAI SDK create() methods to count calls. Idempotent."""
    global _installed
    if _installed:
        return

    try:
        from openai.resources.chat.completions import Completions

        _orig_create = Completions.create

        def _counted_create(self, *args, **kwargs):
            try:
                incr_chat(str(kwargs.get("model", "unknown")))
            except Exception:
                pass
            return _orig_create(self, *args, **kwargs)

        Completions.create = _counted_create
    except Exception:
        pass

    try:
        from openai.resources.embeddings import Embeddings

        _orig_embed = Embeddings.create

        def _counted_embed(self, *args, **kwargs):
            try:
                incr_embeddings(str(kwargs.get("model", "unknown")))
            except Exception:
                pass
            return _orig_embed(self, *args, **kwargs)

        Embeddings.create = _counted_embed
    except Exception:
        pass

    _installed = True
