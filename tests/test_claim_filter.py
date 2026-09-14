import agent.tools.claim_filter as cf
from agent.state import AgentState


def state(answer, articles=None):
    s = AgentState(session_id="s", user_query="q")
    s.final_answer = answer
    s.relevant_articles = articles or []
    return s


def test_strip_disclaimer_block():
    text = "Evidence supports fiber.\n\nDisclaimer: informational purposes only."
    assert cf.strip_disclaimer(text) == "Evidence supports fiber."


def test_strip_gap_talk_closer():
    text = "Omega-3 lowered triglycerides. Further research is needed."
    assert cf.strip_gap_talk_sentences(text) == "Omega-3 lowered triglycerides."


def test_keep_measured_caveat():
    text = "Results were mixed across 5 studies with 200 participants."
    assert cf.strip_gap_talk_sentences(text) == text


def test_split_claims():
    assert cf.split_claims("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_evidence_passages_prefers_answer_body():
    text = cf._evidence_passages([{"title": "T", "abstract": "a", "answer_body": "long answer"}])
    assert "[1] T" in text and "long answer" in text


def test_filter_disabled_strips_meta(monkeypatch):
    monkeypatch.setenv("AGENT_CLAIM_FILTER", "false")
    filtered, meta = cf.apply_claim_filter(state("Finding. Further research is needed."))
    assert filtered == "Finding." and meta["enabled"] is False


def test_no_evidence_keeps_claims(monkeypatch):
    monkeypatch.setenv("AGENT_CLAIM_FILTER", "true")
    filtered, meta = cf.apply_claim_filter(state("Finding one. Finding two."))
    assert filtered == "Finding one. Finding two."
    assert meta["kept_claims"] == 2


def test_supported_claims_only(monkeypatch):
    monkeypatch.setenv("AGENT_CLAIM_FILTER", "true")
    monkeypatch.setattr(cf, "_verify_claims_batch", lambda claims, passages: claims[:1])
    filtered, meta = cf.apply_claim_filter(state("Supported. Invented.", [{"abstract": "Evidence"}]))
    assert filtered == "Supported." and meta["kept_claims"] == 1


def test_zero_supported_refuses(monkeypatch):
    monkeypatch.setattr(cf, "_verify_claims_batch", lambda claims, passages: [])
    filtered, meta = cf.apply_claim_filter(state("Invented.", [{"abstract": "Evidence"}]))
    assert filtered == "" and meta["refuse"] is True
