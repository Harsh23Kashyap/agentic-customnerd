import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import llm_meter


def _usage(prompt, completion, total=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
    )


def test_meter_reset_increment_snapshot():
    llm_meter.reset()
    llm_meter.incr_chat("gpt", 2)
    llm_meter.incr_embeddings("embed", 3)
    snap = llm_meter.snapshot()
    assert snap["chat"] == 2
    assert snap["embeddings"] == 3
    assert snap["total"] == 5
    assert snap["by_model"] == {"gpt": 2}
    assert snap["input_tokens"] == 0
    assert snap["output_tokens"] == 0
    assert snap["total_tokens"] == 0
    assert snap["estimated_cost_usd"] == 0.0


def test_install_idempotent():
    llm_meter.install()
    llm_meter.install()
    assert llm_meter._installed is True


def test_increment_defaults_and_copy():
    llm_meter.reset()
    llm_meter.incr_chat()
    llm_meter.incr_embeddings()
    snap = llm_meter.snapshot()
    snap["by_model"]["mutated"] = 1
    snap["calls"].append({"mutated": True})
    fresh = llm_meter.snapshot()
    assert fresh["chat"] == 1
    assert fresh["embeddings"] == 1
    assert fresh["total"] == 2
    assert fresh["by_model"] == {"unknown": 1}
    assert fresh["calls"] == []


def test_wrap_create_records_mocked_usage():
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args
        assert kwargs["model"] == "gpt-4-turbo"
        return SimpleNamespace(usage=_usage(237, 30, 267))

    wrapped = llm_meter.wrap_create(orig, kind="chat")
    with llm_meter.use_stage("Retrieval"):
        wrapped(object(), model="gpt-4-turbo")

    snap = llm_meter.snapshot()
    assert snap["chat"] == 1
    assert snap["input_tokens"] == 237
    assert snap["output_tokens"] == 30
    assert snap["total_tokens"] == 267
    assert snap["llm_input_tokens"] == 237
    assert snap["estimated_cost_usd"] == 0.00327
    assert snap["usage_missing_calls"] == 0
    assert snap["calls"][0]["stage"] == "Retrieval"
    assert snap["calls"][0]["lifecycle_stage"] == "Retrieval"
    assert snap["by_stage"]["Retrieval"]["calls"] == 1
    assert snap["by_stage"]["Retrieval"]["input_tokens"] == 237
    assert snap["llm_by_stage"]["Relevance"]["calls"] == 0


def test_wrap_create_sums_several_mocked_calls_and_embeddings():
    llm_meter.reset()

    def chat(self, *args, **kwargs):
        del self, args, kwargs
        return {"usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}}

    def embed(self, *args, **kwargs):
        del self, args, kwargs
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=50, total_tokens=50))

    llm_meter.wrap_create(chat, kind="chat")(object(), model="gpt-4-turbo")
    llm_meter.wrap_create(chat, kind="chat")(object(), model="gpt-4-turbo")
    llm_meter.wrap_create(embed, kind="embeddings")(object(), model="text-embedding-3-small")

    snap = llm_meter.snapshot()
    assert snap["chat"] == 2
    assert snap["embeddings"] == 1
    assert snap["total"] == 3
    assert snap["input_tokens"] == 250
    assert snap["output_tokens"] == 20
    assert snap["total_tokens"] == 270
    # 200 * $10/M + 20 * $30/M + 50 * $0.02/M
    assert snap["estimated_cost_usd"] == pytest.approx(0.0026, abs=1e-5)


def test_missing_usage_does_not_invent_token_totals():
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args, kwargs
        return SimpleNamespace(usage=None)

    llm_meter.wrap_create(orig, kind="chat")(object(), model="gpt-4-turbo")
    snap = llm_meter.snapshot()
    assert snap["chat"] == 1
    assert snap["input_tokens"] is None
    assert snap["output_tokens"] is None
    assert snap["total_tokens"] is None
    assert snap["estimated_cost_usd"] is None
    assert snap["usage_missing_calls"] == 1
    assert snap["calls"][0]["input_tokens"] is None


def test_failed_call_is_counted_without_tokens():
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args, kwargs
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        llm_meter.wrap_create(orig, kind="chat")(object(), model="gpt-4-turbo")
    snap = llm_meter.snapshot()
    assert snap["chat"] == 1
    assert snap["total_tokens"] is None


def test_stream_records_usage_from_last_chunk():
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args
        assert kwargs["stream"] is True
        yield SimpleNamespace(usage=None)
        yield SimpleNamespace(usage=_usage(4, 1, 5))

    wrapped = llm_meter.wrap_create(orig, kind="chat")
    chunks = list(wrapped(object(), model="gpt-4o-mini", stream=True))
    assert len(chunks) == 2
    snap = llm_meter.snapshot()
    assert snap["input_tokens"] == 4
    assert snap["output_tokens"] == 1
    assert snap["total_tokens"] == 5
    assert snap["estimated_cost_usd"] == 0.0


def test_unknown_model_keeps_tokens_and_omits_cost():
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args, kwargs
        return SimpleNamespace(usage=_usage(10, 2, 12))

    llm_meter.wrap_create(orig, kind="chat")(object(), model="local-model")
    snap = llm_meter.snapshot()
    assert snap["input_tokens"] == 10
    assert snap["output_tokens"] == 2
    assert snap["total_tokens"] == 12
    assert snap["estimated_cost_usd"] is None


def test_stage_cost_unknown_price_stays_null():
    """A stage that mixes a priced call with an unpriced model keeps tokens and nulls cost."""
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args
        if kwargs["model"] == "gpt-4-turbo":
            return SimpleNamespace(usage=_usage(100, 10, 110))
        return SimpleNamespace(usage=_usage(40, 4, 44))

    wrapped = llm_meter.wrap_create(orig, kind="chat")
    with llm_meter.use_stage("Retrieval"):
        wrapped(object(), model="gpt-4-turbo")
        wrapped(object(), model="local-model")

    snap = llm_meter.snapshot()
    stage = snap["by_stage"]["Retrieval"]
    assert stage["calls"] == 2
    assert stage["input_tokens"] == 140
    assert stage["output_tokens"] == 14
    assert stage["total_tokens"] == 154
    assert stage["estimated_cost_usd"] is None
    assert snap["llm_by_stage"]["Retrieval"]["estimated_cost_usd"] is None
    assert snap["input_tokens"] == 140
    assert snap["total_tokens"] == 154
    assert snap["estimated_cost_usd"] is None


def test_stage_inferred_from_caller_name():
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args, kwargs
        return SimpleNamespace(usage=_usage(3, 1, 4))

    wrapped = llm_meter.wrap_create(orig, kind="chat")

    def relevance_classifier():
        wrapped(object(), model="gpt-4-turbo")

    relevance_classifier()
    assert llm_meter.snapshot()["calls"][0]["stage"] == "Relevance"


def test_price_matches_exact_name_or_dated_suffix_only():
    """gpt-4o-2024-08-06 stays on gpt-4o pricing; gpt-4o2 does not inherit it."""
    assert llm_meter.estimate_cost_usd("gpt-4o", 1_000_000, 0) == 2.5
    assert llm_meter.estimate_cost_usd("GPT-4o-2024-08-06", 1_000_000, 0) == 2.5
    assert llm_meter.estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert llm_meter.estimate_cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15
    assert llm_meter.estimate_cost_usd("gpt-4-turbo", 1_000_000, 0) == 10.0
    assert llm_meter.estimate_cost_usd("gpt-4-turbo-preview", 1_000_000, 0) == 10.0
    assert llm_meter.estimate_cost_usd("gpt-4-turbo-2024-04-09", 1_000_000, 0) == 10.0
    assert llm_meter.estimate_cost_usd("gpt-4", 1_000_000, 0) == 30.0
    assert llm_meter.estimate_cost_usd("gpt-4-0613", 1_000_000, 0) == 30.0
    assert llm_meter.estimate_cost_usd("text-embedding-3-small", 1_000_000, 0) == 0.02
    assert llm_meter.estimate_cost_usd("gpt-4o2", 1_000_000, 0) is None
    assert llm_meter.estimate_cost_usd("gpt-4oextra", 1_000_000, 0) is None
    assert llm_meter.estimate_cost_usd("unknown", 1_000_000, 0) is None


def _legacy_infer_stage():
    """Previous inference: full inspect.stack(), skip this meter file, first hint wins."""
    names = []
    for frame in inspect.stack():
        try:
            filename = frame.filename or ""
            if filename and Path(filename).resolve() == llm_meter._METER_FILE:
                continue
        except OSError:
            pass
        names.append(frame.function or "")
        names.append(frame.filename or "")
    return llm_meter.infer_stage(names)


def test_caller_walk_matches_full_stack_inference():
    """A bounded walk must infer the same stage as the old full stack walk."""

    def claim_filter_helper():
        def inner():
            walked = llm_meter.infer_stage()
            legacy = _legacy_infer_stage()
            return walked, legacy

        return inner()

    assert claim_filter_helper() == ("Claim Filter", "Claim Filter")


def test_stage_cache_follows_the_outer_caller_not_the_shared_helper():
    def inner():
        return llm_meter.infer_stage(), _legacy_infer_stage()

    def tool_search():
        return inner()

    def summarize_relevant():
        return inner()

    assert tool_search() == ("Retrieval", "Retrieval")
    assert tool_search() == ("Retrieval", "Retrieval")
    assert summarize_relevant() == ("Organize", "Organize")


def test_telemetry_csv_and_summary_from_mocked_usage(tmp_path):
    llm_meter.reset()

    def orig(self, *args, **kwargs):
        del self, args, kwargs
        return SimpleNamespace(usage=_usage(1000, 40, 1040))

    llm_meter.wrap_create(orig, kind="chat")(object(), model="gpt-4-turbo")
    first = {"question_id": 1, "duration_seconds": 10, "llm_calls": llm_meter.snapshot(), "ragas_faithfulness": 1.0}
    llm_meter.reset()
    llm_meter.wrap_create(orig, kind="chat")(object(), model="gpt-4-turbo")
    llm_meter.wrap_create(orig, kind="chat")(object(), model="gpt-4-turbo")
    second = {"question_id": 2, "duration_seconds": 5.5, "llm_calls": llm_meter.snapshot(), "sts_score": 0.5}

    csv_path = tmp_path / "telemetry.csv"
    summary_path = tmp_path / "telemetry_summary.json"
    summary = llm_meter.write_telemetry(
        [first, second],
        csv_path,
        summary_path,
        run_id="mocked",
        agent_commit="abc",
    )
    text = csv_path.read_text(encoding="utf-8")
    assert "llm_input_tokens" in text.splitlines()[0]
    assert "1000" in text
    assert "2000" in text
    assert summary["n_questions"] == 2
    assert summary["llm_input_tokens_sum"] == 3000
    assert summary["llm_output_tokens_sum"] == 120
    assert summary["llm_total_tokens_sum"] == 3120
    assert summary["llm_calls_total_sum"] == 3
    assert summary["duration_seconds_sum"] == 15.5
    assert summary["estimated_cost_usd_sum"] == pytest.approx(0.0336, abs=1e-5)
    assert "Real provider usage" in summary["token_fields_note"]
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["llm_total_tokens_sum"] == 3120
    assert saved["agent_commit"] == "abc"


def test_summary_stays_null_when_any_question_lacks_tokens():
    rows = [
        {"question_id": 1, "llm_calls": {"total": 1, "chat": 1, "embeddings": 0, "input_tokens": 10, "output_tokens": 1, "total_tokens": 11, "estimated_cost_usd": 0.00013}},
        {"question_id": 2, "llm_calls": {"total": 1, "chat": 1, "embeddings": 0, "input_tokens": None, "output_tokens": None, "total_tokens": None}},
    ]
    summary = llm_meter.summarize_telemetry(rows, run_id="partial", agent_commit="abc")
    assert summary["llm_input_tokens_sum"] is None
    assert summary["llm_total_tokens_sum"] is None
    assert summary["estimated_cost_usd_sum"] is None
    assert "missing" in summary["token_fields_note"]


@pytest.mark.parametrize("mode", ["early", "exhausted", "separate_iterator", "close_error"])
def test_stream_closes_provider_once_and_records_usage(mode):
    llm_meter.reset()

    class Stream:
        def __init__(self):
            self.close_calls = 0
            self.chunks = iter([SimpleNamespace(usage=_usage(4, 1, 5))])

        def __iter__(self):
            return self.chunks if mode == "separate_iterator" else self

        def __next__(self):
            return next(self.chunks)

        def close(self):
            self.close_calls += 1
            if mode == "close_error":
                raise RuntimeError("close failed")

    stream = Stream()
    wrapped = llm_meter.wrap_create(lambda self, **kwargs: stream, kind="chat")
    result = wrapped(object(), model="gpt-4o-mini", stream=True)
    if mode == "exhausted":
        list(result)
    else:
        next(result)
    if mode == "close_error":
        with pytest.raises(RuntimeError, match="close failed"):
            result.close()
    else:
        result.close()
    result.close()
    assert stream.close_calls == 1
    snap = llm_meter.snapshot()
    assert snap["chat"] == 1
    assert snap["total_tokens"] == 5
