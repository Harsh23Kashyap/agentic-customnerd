from agent.state import AgentState
import agent.tools.search_focus as sf


def make(query="AWS Docker timeout", title=None):
    return AgentState(session_id="s", user_query=query, query_title=title)


def test_provider_detection_dedupes_aliases():
    assert sf._detect_providers("AWS amazon S3 and k8s kubernetes") == ["AWS", "Kubernetes"]


def test_tier_ladder_clamps():
    assert sf.tier_for_search_attempt(1) == "strict"
    assert sf.tier_for_search_attempt(2) == "extended_keywords"
    assert sf.tier_for_search_attempt(99) == "tag_similar"


def test_long_body_reverses_ladder():
    state = make("x" * 1501)
    assert sf.tier_for_search_attempt(1, state=state) == "tag_similar"


def test_build_focus_includes_title_provider_and_reason(monkeypatch):
    monkeypatch.setattr(sf, "extract_key_terms", lambda q: {"docker", "timeout"})
    focus = sf.build_search_focus(make("AWS docker timeout", "Container timeout"), reason="zero_retrieval")
    assert "Container timeout" in focus
    assert "AWS" in focus
    assert "stack overflow solution" in focus


def test_build_focus_bounded(monkeypatch):
    monkeypatch.setattr(sf, "extract_key_terms", lambda q: {"x" * 100})
    assert len(sf.build_search_focus(make("q" * 500))) <= 400


def test_hint_deterministic(monkeypatch):
    monkeypatch.setenv("AGENT_DETERMINISTIC_LADDER", "true")
    state = make(); state.search_attempts = 2
    assert sf.suggest_tier_hint(state, reason="zero_retrieval") == "extended_keywords"
