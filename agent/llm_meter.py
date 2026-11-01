"""Thread-safe per-request LLM call meter.

Patches the OpenAI SDK at the class level so every chat-completion and
embedding call (legacy backend + agent tools + orchestrator) records provider
usage: input, output, and total tokens, plus cost when the model price is known.

The counter is a process-global reset at the start of each request. This is
correct as long as requests are processed sequentially (as the eval harness
does). Concurrent requests would share the counter.
"""

from __future__ import annotations

import inspect
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Optional

_METER_FILE = Path(__file__).resolve()

_lock = threading.Lock()
_stage_var: ContextVar[Optional[str]] = ContextVar("llm_meter_stage", default=None)

# USD per 1M tokens: (input, output). Longest prefix wins.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
}

_CANONICAL_STAGES = (
    "Retrieval",
    "Relevance",
    "Final Answer",
    "Claim Filter",
    "Organize",
    "Assess",
    "Verify",
)

# First match wins, so more specific names stay above broader ones.
_STAGE_HINTS = (
    ("_verify_claims", "Claim Filter"),
    ("apply_claim_filter", "Claim Filter"),
    ("claim_filter", "Claim Filter"),
    ("verify_faithfulness", "Verify"),
    ("faithfulness", "Verify"),
    ("two_pass", "Verify"),
    ("relevance", "Relevance"),
    ("generate_final", "Final Answer"),
    ("grounded_fallback", "Final Answer"),
    ("synthesize", "Final Answer"),
    ("_summarize_one", "Organize"),
    ("summarize_relevant", "Organize"),
    ("organize", "Organize"),
    ("assess_retrieval", "Assess"),
    ("tool_assess", "Assess"),
    ("query_generation", "Retrieval"),
    ("tool_search", "Retrieval"),
    ("collect_articles", "Retrieval"),
)

_installed = False


def _blank_state() -> dict:
    return {
        "chat": 0,
        "embeddings": 0,
        "by_model": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "latency": 0.0,
        "usage_missing": 0,
        "cost_complete": True,
        "calls": [],
        "wall": {},
    }


_state = _blank_state()


def reset() -> None:
    with _lock:
        _state.clear()
        _state.update(_blank_state())


def incr_chat(model: str = "unknown", n: int = 1) -> None:
    """Count chat calls that do not carry provider usage (legacy helper)."""
    with _lock:
        _state["chat"] += n
        _state["by_model"][model] = _state["by_model"].get(model, 0) + n


def incr_embeddings(model: str = "unknown", n: int = 1) -> None:
    with _lock:
        _state["embeddings"] += n


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Return rounded USD cost, or None when the model price is unknown."""
    price = _price_for(model)
    if price is None:
        return None
    inp, out = price
    return round((input_tokens * inp + output_tokens * out) / 1_000_000, 5)


def _price_for(model: str) -> Optional[tuple[float, float]]:
    """Return the price for an exact model id or a dated snapshot of one.

    Longest key wins, so gpt-4o-mini is not priced as gpt-4o. A snapshot such
    as gpt-4o-2024-08-06 matches ``key + "-"``. A bare prefix does not: a name
    like gpt-4o2 must not inherit gpt-4o pricing.
    """
    name = (model or "").strip().lower()
    if not name or name == "unknown":
        return None
    for key in sorted(_PRICES, key=len, reverse=True):
        if name == key or name.startswith(key + "-"):
            return _PRICES[key]
    return None


# (filename, function name, firstlineno) through the frame that decided the
# stage. Only decisive matches are stored, so a shared helper is not pinned to
# the first caller that used it.
_STAGE_CACHE: dict[tuple[tuple[str, str, int], ...], str] = {}
_STAGE_CACHE_MAX = 128
_METER_FILENAME_CACHE: dict[str, bool] = {str(_METER_FILE): True}


def _is_meter_filename(filename: str) -> bool:
    """True when this frame is agent/llm_meter.py. Avoids a stat on every other file."""
    if not filename:
        return False
    cached = _METER_FILENAME_CACHE.get(filename)
    if cached is not None:
        return cached
    hit = False
    normalized = filename.replace("\\", "/")
    if normalized.endswith("/llm_meter.py") or normalized == "llm_meter.py":
        try:
            hit = Path(filename).resolve() == _METER_FILE
        except OSError:
            hit = False
    _METER_FILENAME_CACHE[filename] = hit
    return hit


def _stage_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    for needle, stage in _STAGE_HINTS:
        if needle in text:
            return stage
    return None


def _remember_stage(key: tuple[tuple[str, str, int], ...], stage: str) -> None:
    if key in _STAGE_CACHE:
        return
    if len(_STAGE_CACHE) >= _STAGE_CACHE_MAX:
        _STAGE_CACHE.clear()
    _STAGE_CACHE[key] = stage


def _stage_from_names(names: list[str]) -> str:
    for blob in names:
        stage = _stage_in_text(blob or "")
        if stage:
            return stage
    return "Other"


def _stage_from_caller() -> str:
    """Infer a stage from the caller chain without reading source lines.

    ``inspect.stack()`` formats every frame (including a source read) even
    when the first non-meter caller already names the stage. Walking
    ``f_back`` and stopping at the first hint uses the same function name and
    filename, in the same inner-to-outer order, and skips this module the
    same way. The cache remembers a chain that already decided a stage so a
    tight loop does not rescan hints; it does not cache "Other", because a
    shorter chain can be a prefix of a later, decisive one.
    """
    frame = inspect.currentframe()
    key: list[tuple[str, str, int]] = []
    try:
        while frame is not None:
            code = frame.f_code
            filename = code.co_filename or ""
            function = code.co_name or ""
            firstlineno = int(code.co_firstlineno)
            frame = frame.f_back
            if _is_meter_filename(filename):
                continue
            key.append((filename, function, firstlineno))
            cache_key = tuple(key)
            cached = _STAGE_CACHE.get(cache_key)
            if cached is not None:
                return cached
            stage = _stage_in_text(function) or _stage_in_text(filename)
            if stage is not None:
                _remember_stage(cache_key, stage)
                return stage
    finally:
        del frame
    return "Other"


def infer_stage(names: Optional[list[str]] = None) -> str:
    """Map caller function / filename text to a pipeline stage."""
    explicit = _stage_var.get()
    if explicit:
        return explicit
    if names is None:
        return _stage_from_caller()
    return _stage_from_names(names)


@contextmanager
def use_stage(name: str):
    """Label LLM calls made on this thread until the block exits."""
    token = _stage_var.set(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        _stage_var.reset(token)
        with _lock:
            _state["wall"][name] = _state["wall"].get(name, 0.0) + elapsed


def staged(name: str):
    """Decorator form of use_stage."""

    def deco(fn):
        def wrapper(*args, **kwargs):
            with use_stage(name):
                return fn(*args, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        wrapper.__wrapped__ = fn
        return wrapper

    return deco


def _usage_triplet(usage: Any) -> Optional[tuple[int, int, int]]:
    if usage is None:
        return None

    def _get(*names: str) -> Optional[int]:
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if value is not None:
                return int(value)
        return None

    prompt = _get("prompt_tokens", "input_tokens")
    completion = _get("completion_tokens", "output_tokens")
    total = _get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    prompt = 0 if prompt is None else prompt
    completion = 0 if completion is None else completion
    if total is None:
        total = prompt + completion
    return prompt, completion, total


def _extract_usage(response: Any) -> Optional[tuple[int, int, int]]:
    if response is None:
        return None
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    return _usage_triplet(usage)


def record_call(
    *,
    kind: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    latency_seconds: float = 0.0,
    stage: Optional[str] = None,
) -> None:
    """Record one provider call. Token fields stay None when usage was absent."""
    stage_name = stage or infer_stage()
    cost = None
    if input_tokens is not None and output_tokens is not None:
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
    detail = {
        "timestamp": time.time(),
        "provider": "openai",
        "model": model or "unknown",
        "kind": kind,
        "stage": stage_name,
        "lifecycle_stage": stage_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(float(latency_seconds or 0.0), 4),
        "estimated_cost_usd": cost,
    }
    with _lock:
        if kind == "embeddings":
            _state["embeddings"] += 1
        else:
            _state["chat"] += 1
            _state["by_model"][model or "unknown"] = _state["by_model"].get(model or "unknown", 0) + 1
        if input_tokens is None or output_tokens is None or total_tokens is None:
            _state["usage_missing"] += 1
            _state["cost_complete"] = False
        else:
            _state["input_tokens"] += int(input_tokens)
            _state["output_tokens"] += int(output_tokens)
            _state["total_tokens"] += int(total_tokens)
            if cost is None:
                _state["cost_complete"] = False
            else:
                _state["cost"] = round(_state["cost"] + cost, 5)
        _state["latency"] += float(latency_seconds or 0.0)
        _state["calls"].append(detail)


def _empty_stage() -> dict:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "llm_latency_seconds": 0.0,
        "wall_seconds": 0.0,
    }


def _stage_rollup(calls: list[dict], wall: dict) -> dict:
    rollup: dict[str, dict] = {name: _empty_stage() for name in _CANONICAL_STAGES}
    for name, seconds in wall.items():
        bucket = rollup.setdefault(name, _empty_stage())
        bucket["wall_seconds"] = round(float(seconds), 4)
    incomplete: set[str] = set()
    unpriced: set[str] = set()
    for call in calls:
        name = call.get("stage") or "Other"
        bucket = rollup.setdefault(name, _empty_stage())
        bucket["calls"] += 1
        bucket["llm_latency_seconds"] = round(
            bucket["llm_latency_seconds"] + float(call.get("latency_seconds") or 0.0), 4
        )
        if call.get("input_tokens") is None:
            incomplete.add(name)
            continue
        bucket["input_tokens"] += int(call["input_tokens"])
        bucket["output_tokens"] += int(call["output_tokens"] or 0)
        bucket["total_tokens"] += int(call["total_tokens"] or 0)
        # An unpriced model must not look like a finished stage total.
        # Keep the known token counts; only the cost becomes null.
        if call.get("estimated_cost_usd") is None:
            unpriced.add(name)
        else:
            bucket["estimated_cost_usd"] = round(
                bucket["estimated_cost_usd"] + float(call["estimated_cost_usd"]), 5
            )
    for name in incomplete:
        rollup[name]["input_tokens"] = None
        rollup[name]["output_tokens"] = None
        rollup[name]["total_tokens"] = None
        rollup[name]["estimated_cost_usd"] = None
    for name in unpriced - incomplete:
        rollup[name]["estimated_cost_usd"] = None
    return rollup


def snapshot() -> dict:
    with _lock:
        calls = [dict(item) for item in _state["calls"]]
        wall = dict(_state["wall"])
        by_model = dict(_state["by_model"])
        chat = _state["chat"]
        embeddings = _state["embeddings"]
        usage_missing = _state["usage_missing"]
        cost_complete = _state["cost_complete"]
        input_tokens = _state["input_tokens"]
        output_tokens = _state["output_tokens"]
        total_tokens = _state["total_tokens"]
        cost = _state["cost"]
        latency = _state["latency"]

    complete = usage_missing == 0
    input_out = input_tokens if complete else None
    output_out = output_tokens if complete else None
    total_out = total_tokens if complete else None
    cost_out = round(cost, 5) if cost_complete and complete else None
    by_stage = _stage_rollup(calls, wall)
    return {
        "chat": chat,
        "embeddings": embeddings,
        "total": chat + embeddings,
        "by_model": by_model,
        "input_tokens": input_out,
        "output_tokens": output_out,
        "total_tokens": total_out,
        "llm_input_tokens": input_out,
        "llm_output_tokens": output_out,
        "llm_total_tokens": total_out,
        "estimated_cost_usd": cost_out,
        "llm_latency_seconds": round(latency, 4),
        "usage_missing_calls": usage_missing,
        "by_stage": by_stage,
        "llm_by_stage": by_stage,
        "calls": calls,
        "llm_call_details": calls,
    }


def _is_stream(result: Any, kwargs: dict) -> bool:
    if kwargs.get("stream") is not True:
        return False
    return hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict, list))


class _UsageIterator:
    """Pull usage off the last stream chunk, then record the call."""

    def __init__(self, inner: Any, finish) -> None:
        self._source = inner
        self._inner = iter(inner)
        self._finish = finish
        self._usage = None
        self._done = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._inner)
        except StopIteration:
            self._close()
            raise
        usage = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
        if usage is not None:
            self._usage = usage
        return chunk

    def _close(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            close = getattr(self._source, "close", None)
            if close is not None:
                close()
        finally:
            self._finish(self._usage)

    def close(self) -> None:
        self._close()


def wrap_create(orig, *, kind: str = "chat"):
    """Wrap an OpenAI create() so the returned response's usage is recorded."""

    def _counted(self, *args, **kwargs):
        model = str(kwargs.get("model") or "unknown")
        stage = infer_stage()
        started = time.perf_counter()

        def _finish(usage: Any) -> None:
            triplet = _usage_triplet(usage)
            latency = time.perf_counter() - started
            if triplet is None:
                record_call(
                    kind=kind,
                    model=model,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    latency_seconds=latency,
                    stage=stage,
                )
                return
            record_call(
                kind=kind,
                model=model,
                input_tokens=triplet[0],
                output_tokens=triplet[1],
                total_tokens=triplet[2],
                latency_seconds=latency,
                stage=stage,
            )

        try:
            result = orig(self, *args, **kwargs)
        except Exception:
            record_call(
                kind=kind,
                model=model,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_seconds=time.perf_counter() - started,
                stage=stage,
            )
            raise

        if _is_stream(result, kwargs):
            return _UsageIterator(result, _finish)

        _finish(result.get("usage") if isinstance(result, dict) else getattr(result, "usage", None))
        return result

    return _counted


def install() -> None:
    """Monkeypatch OpenAI SDK create() methods to record usage. Idempotent."""
    global _installed
    if _installed:
        return

    try:
        from openai.resources.chat.completions import Completions

        Completions.create = wrap_create(Completions.create, kind="chat")
    except Exception:
        pass

    try:
        from openai.resources.embeddings import Embeddings

        Embeddings.create = wrap_create(Embeddings.create, kind="embeddings")
    except Exception:
        pass

    _installed = True


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def question_telemetry(record: dict) -> dict:
    """One eval row -> telemetry columns. Tokens come from llm_calls usage."""
    calls = record.get("llm_calls") if isinstance(record.get("llm_calls"), dict) else {}

    def pick(*keys: str):
        for source in (record, calls):
            if not isinstance(source, dict):
                continue
            for key in keys:
                if key in source and source[key] is not None and source[key] != "":
                    return source[key]
        return None

    return {
        "question_id": record.get("question_id"),
        "duration_seconds": pick("duration_seconds"),
        "llm_calls_total": pick("llm_calls_total", "total"),
        "llm_calls_chat": pick("llm_calls_chat", "chat"),
        "llm_calls_embeddings": pick("llm_calls_embeddings", "embeddings"),
        "llm_input_tokens": pick("llm_input_tokens", "input_tokens"),
        "llm_output_tokens": pick("llm_output_tokens", "output_tokens"),
        "llm_total_tokens": pick("llm_total_tokens", "total_tokens"),
        "estimated_cost_usd": pick("estimated_cost_usd"),
        "ragas_faithfulness": record.get("ragas_faithfulness"),
        "ragas_answer_relevancy": record.get("ragas_answer_relevancy"),
        "ragas_answer_similarity": record.get("ragas_answer_similarity"),
        "sts_score": record.get("sts_score"),
    }


TELEMETRY_FIELDS = (
    "question_id",
    "duration_seconds",
    "llm_calls_total",
    "llm_calls_chat",
    "llm_calls_embeddings",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_total_tokens",
    "estimated_cost_usd",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_answer_similarity",
    "sts_score",
)


def _sum_field(rows: list[dict], key: str):
    values = [_num(row.get(key)) for row in rows]
    if any(value is None for value in values):
        return None
    if key in {"llm_calls_total", "llm_calls_chat", "llm_calls_embeddings", "llm_input_tokens", "llm_output_tokens", "llm_total_tokens"}:
        return int(sum(values))
    return round(sum(values), 5)


def summarize_telemetry(records: list[dict], *, run_id: str, agent_commit: Optional[str] = None) -> dict:
    """Aggregate per-question telemetry into a run summary. Missing tokens stay null."""
    rows = [question_telemetry(record) for record in records]
    n = len(rows)
    duration_sum = _sum_field(rows, "duration_seconds")
    input_sum = _sum_field(rows, "llm_input_tokens")
    output_sum = _sum_field(rows, "llm_output_tokens")
    total_sum = _sum_field(rows, "llm_total_tokens")
    cost_sum = _sum_field(rows, "estimated_cost_usd")
    missing = input_sum is None or output_sum is None or total_sum is None
    return {
        "run_id": run_id,
        "agent_commit": agent_commit,
        "n_questions": n,
        "duration_seconds_sum": duration_sum,
        "duration_seconds_mean": round(duration_sum / n, 2) if duration_sum is not None and n else None,
        "llm_calls_total_sum": _sum_field(rows, "llm_calls_total"),
        "llm_input_tokens_sum": input_sum,
        "llm_output_tokens_sum": output_sum,
        "llm_total_tokens_sum": total_sum,
        "estimated_cost_usd_sum": cost_sum,
        "token_fields_note": (
            "Real provider usage. At least one question is missing input/output/total tokens."
            if missing
            else "Real provider usage recorded on each LLM call (input, output, total, and cost when priced)."
        ),
        "questions": rows,
    }


def write_telemetry(records: list[dict], csv_path, summary_path, *, run_id: str, agent_commit: Optional[str] = None) -> dict:
    """Write the per-question telemetry CSV and the run summary JSON."""
    import csv
    import json
    from pathlib import Path

    rows = [question_telemetry(record) for record in records]
    summary = summarize_telemetry(records, run_id=run_id, agent_commit=agent_commit)
    summary_out = {key: value for key, value in summary.items() if key != "questions"}
    dest_csv = Path(csv_path)
    dest_summary = Path(summary_path)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TELEMETRY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in TELEMETRY_FIELDS})
    dest_summary.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    return summary_out
