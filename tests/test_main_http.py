"""HTTP-layer tests for main.py with the pipeline, legacy backend and thread pool stubbed.

No model call, no network and no sibling customnerd-backend is needed.
"""
import importlib
import queue
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def web(monkeypatch, tmp_path):
    import bridge.legacy as legacy
    from agent import llm_meter
    monkeypatch.setattr(legacy, "ensure_legacy_backend", lambda *a, **k: tmp_path)
    monkeypatch.setattr(llm_meter, "install", lambda *a, **k: None)
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    submitted = []

    class Pool:
        def submit(self, fn, *args):
            submitted.append((fn, args))

    monkeypatch.setattr(main, "executor", Pool())
    main.update_queues.clear()
    yield main, TestClient(main.app), submitted
    main.update_queues.clear()
    sys.modules.pop("main", None)


def test_blank_query_is_refused_without_a_session_or_work(web):
    main, client, submitted = web
    for blank in ("", " ", "\n\t "):
        r = client.post("/agent/query", data={"user_query": blank})
        assert r.status_code == 422, blank
        assert "blank" in r.json()["detail"]
    assert submitted == [] and main.update_queues == {}


def test_blank_query_is_refused_on_the_compatible_endpoint_too(web):
    main, client, submitted = web
    r = client.post("/process_detailed_combined_query", data={"user_query": "   "})
    assert r.status_code == 422
    assert submitted == [] and main.update_queues == {}


def test_missing_query_is_refused(web):
    _, client, submitted = web
    assert client.post("/agent/query", data={}).status_code == 422
    assert submitted == []


def test_real_query_starts_one_session_and_passes_fields_through(web):
    main, client, submitted = web
    r = client.post("/agent/query", data={"user_query": "  how do I x?  ", "max_date": "2024/01/01",
                                          "query_title": "T"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert isinstance(main.update_queues[sid], queue.Queue)
    assert len(submitted) == 1
    fn, args = submitted[0]
    assert fn is main._run_agent_logic
    assert args == ("  how do I x?  ", sid, "2024/01/01", "T")


def test_each_query_gets_its_own_session(web):
    main, client, submitted = web
    a = client.post("/agent/query", data={"user_query": "one"}).json()["session_id"]
    b = client.post("/agent/query", data={"user_query": "two"}).json()["session_id"]
    assert a != b and set(main.update_queues) == {a, b} and len(submitted) == 2


def test_compatible_endpoint_matches_agent_endpoint(web):
    main, client, submitted = web
    r = client.post("/process_detailed_combined_query", data={"user_query": "q"})
    assert r.status_code == 200 and len(submitted) == 1
    assert submitted[0][1][0] == "q"


def test_sse_requires_a_session_id(web):
    _, client, _ = web
    assert client.get("/sse").status_code == 400


def test_load_domain_unknown_domain_is_404_and_changes_nothing(web, monkeypatch):
    main, client, _ = web
    import bridge.domain as domain
    calls = []
    monkeypatch.setattr(domain, "list_domains", lambda: ["CloudNerd"])
    monkeypatch.setattr(domain, "activate_domain", lambda *a, **k: calls.append(a))
    r = client.post("/agent/load_domain", data={"domain": "Nope"})
    assert r.status_code == 404 and "CloudNerd" in r.json()["detail"]
    assert calls == []


def test_load_domain_busy_runtime_is_503(web, monkeypatch):
    main, client, _ = web
    import bridge.domain as domain
    def busy(*a, **k):
        raise main.LegacyRuntimeBusy("busy")
    monkeypatch.setattr(domain, "list_domains", lambda: ["CloudNerd"])
    monkeypatch.setattr(domain, "activate_domain", busy)
    assert client.post("/agent/load_domain", data={"domain": "CloudNerd"}).status_code == 503


def test_load_domain_success_returns_status(web, monkeypatch):
    _, client, _ = web
    import bridge.domain as domain
    monkeypatch.setattr(domain, "list_domains", lambda: ["CloudNerd"])
    monkeypatch.setattr(domain, "activate_domain", lambda d, force=False: {"domain": d, "forced": force})
    r = client.post("/agent/load_domain", data={"domain": "CloudNerd"})
    assert r.json() == {"status": "ok", "domain": "CloudNerd", "forced": True}


def test_health_is_degraded_without_keys_and_ok_with_them(web, monkeypatch):
    _, client, _ = web
    import bridge.domain as domain
    monkeypatch.setattr(domain, "get_domain", lambda: "CloudNerd")
    monkeypatch.setattr(domain, "is_dietnerd", lambda: False)
    monkeypatch.setattr(domain, "list_domains", lambda: ["CloudNerd"])
    for k in ("OPENAI_API_KEY", "STACK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    body = client.get("/health").json()
    assert body["status"] == "degraded" and len(body["warnings"]) == 2
    monkeypatch.setenv("OPENAI_API_KEY", "x"); monkeypatch.setenv("STACK_API_KEY", "x")
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["warnings"] == []


def test_health_for_dietnerd_warns_about_pubmed_settings_not_stack(web, monkeypatch):
    _, client, _ = web
    import bridge.domain as domain
    monkeypatch.setattr(domain, "get_domain", lambda: "DietNerd")
    monkeypatch.setattr(domain, "is_dietnerd", lambda: True)
    monkeypatch.setattr(domain, "list_domains", lambda: ["DietNerd"])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    for k in ("ENTREZ_EMAIL", "NCBI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    warnings = client.get("/health").json()["warnings"]
    assert any("ENTREZ_EMAIL" in w for w in warnings) and any("NCBI_API_KEY" in w for w in warnings)
    assert not any("STACK" in w for w in warnings)
