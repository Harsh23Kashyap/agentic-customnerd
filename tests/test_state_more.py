from agent.state import AgentState


def test_articles_for_classify_prefers_organized():
    s=AgentState('s','q'); s.raw_articles=[{'r':1}]; assert s.articles_for_classify==s.raw_articles
    s.organized_articles=[{'o':1}]; assert s.articles_for_classify==s.organized_articles


def test_clear_for_rescue_resets_transient_state():
    s=AgentState('s','q'); s.done=True; s.final_answer='x'; s.citations=['c']; s.raw_articles=[{}]; s.organized_articles=[{}]; s.relevant_articles=[{}]; s.reflect_failed=True; s.cached_evidence_facts='f'; s.cached_fact_overlap=.5; s.faith_regenerate_attempts=2; s.domain_overlap=.1; s.suppress_gap_last=True; s.use_combined_lite=True; s.search_attempts=2
    s.clear_for_rescue_pass()
    assert not s.done and s.final_answer is None and not s.raw_articles and not s.citations
    assert s.search_attempts==0 and s.domain_overlap is None and not s.use_combined_lite
