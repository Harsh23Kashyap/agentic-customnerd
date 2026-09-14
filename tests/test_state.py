from agent.state import AgentState


def test_state_defaults():
    state = AgentState(session_id="s", user_query="q")
    assert state.session_id == "s"
    assert state.user_query == "q"
    assert state.raw_articles == []
    assert state.relevant_articles == []
    assert state.search_attempts == 0


def test_log_step_appends_trace():
    state = AgentState(session_id="s", user_query="q")
    state.log_step("search", {"count": 2})
    assert state.agent_trace == [{"step": "search", "detail": {"count": 2}}]


def test_mutable_defaults_are_isolated():
    one = AgentState(session_id="1", user_query="q")
    two = AgentState(session_id="2", user_query="q")
    one.raw_articles.append({"id": 1})
    assert two.raw_articles == []
