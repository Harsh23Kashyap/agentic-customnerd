import agent.faith_policy as fp
from agent.state import AgentState


def state(query="omega benefits"):
    return AgentState(session_id="s", user_query=query)


def test_env_parsers(monkeypatch):
    monkeypatch.setenv("BOOL", "yes"); monkeypatch.setenv("FLOAT", "bad"); monkeypatch.setenv("INT", "7")
    assert fp._env_bool("BOOL", False) is True
    assert fp._env_float("FLOAT", 1.5) == 1.5
    assert fp._env_int("INT", 2) == 7


def test_post_filter_default_tracks_faith(monkeypatch):
    monkeypatch.delenv("AGENT_POST_FILTER_MODE", raising=False)
    monkeypatch.setenv("AGENT_FAITH_FIRST", "true")
    assert fp.post_filter_mode() == "rank_only"
    monkeypatch.setenv("AGENT_FAITH_FIRST", "false")
    assert fp.post_filter_mode() == "filter"


def test_low_domain_overlap_boundaries(monkeypatch):
    monkeypatch.setenv("AGENT_LOW_DOMAIN_OVERLAP_RESEARCH", "0.15")
    assert fp.is_low_domain_overlap(None) is False
    assert fp.is_low_domain_overlap(.14) is True
    assert fp.is_low_domain_overlap(.15) is False


def test_prepare_combined_context(monkeypatch):
    s = state(); s.relevant_articles = [{"abstract": "omega benefits"}]; s.domain_overlap = .5
    monkeypatch.setenv("AGENT_PREFER_COMBINED_SYNTHESIS", "true")
    fp.prepare_synthesis_context(s)
    assert s.use_combined_lite is True


def test_strip_hedge_sentences():
    assert fp.strip_hedge_sentences("Use this setting. You may need to verify it.") == "Use this setting."


def test_evidence_text_uses_first_available_field():
    s = state(); s.relevant_articles = [{"abstract": "A", "body": "B"}, {"body": "C"}]
    assert fp._cheap_evidence_text(s) == "A\n\nC"


def test_best_overlapping_sentence_prefers_query_match():
    text = "This is a long sentence about unrelated database configuration and caching. Omega supplements lowered triglycerides in a randomized human trial."
    assert "Omega" in fp._best_overlapping_sentence(text, "omega triglycerides")


def test_fallback_passages_are_numbered():
    out = fp._fallback_passages([{"title": "Study", "abstract": "Evidence"}])
    assert out == "[1] Study\nEvidence"


def test_heuristic_fallback_is_question_shaped():
    answer = fp._heuristic_grounded_fallback("Does omega help?", [{"title": "Omega", "abstract": "Omega supplementation lowered triglycerides in randomized adult clinical trials."}])
    assert answer.startswith("Answer for: Does omega help?")
    assert "[1]" in answer


def test_llm_fallback_guard_rejects_uncited():
    assert fp._fallback_looks_extractive("A claim without refs", "A claim appears in the passage") is False


def test_grounded_fallback_no_articles():
    assert fp.build_grounded_fallback_answer(state()) == "No Stack Overflow posts were retrieved for this question."


def test_finalize_passed_keeps_answer():
    s = state(); s.final_answer = "Answer"
    assert fp.finalize_or_refuse_answer(s, {"passed": True}) is False
    assert s.final_answer == "Answer"


def test_finalize_waits_for_rescue():
    s = state(); s.final_answer = "Ungrounded"
    assert fp.finalize_or_refuse_answer(s, {"passed": False, "evidence_overlap": 0, "answer_fact_overlap": 0}) is False


def test_finalize_replaces_after_rescue(monkeypatch):
    s = state(); s.final_answer = "Bad"; s.raw_articles = [{"title": "Study", "abstract": "Omega supplementation lowered triglycerides in randomized adult clinical trials."}]
    monkeypatch.setattr(fp, "ensure_citations_on_refusal", lambda state: state.citations.append("source"))
    assert fp.finalize_or_refuse_answer(s, {"passed": False, "evidence_overlap": 0, "answer_fact_overlap": 0}, overlap_rescue_attempted=True) is True
    assert s.done is True and s.citations == ["source"] and "[1]" in s.final_answer
