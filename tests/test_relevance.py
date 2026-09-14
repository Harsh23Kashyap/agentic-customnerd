import agent.tools.relevance as rel


def article(title="", body="", pmid=None, answer_id=None):
    out = {"title": title, "answer_body": body}
    if pmid is not None: out["PMID"] = pmid
    if answer_id is not None: out["answer_id"] = answer_id
    return out


def test_extract_terms_drops_stopwords_and_code_noise(monkeypatch):
    monkeypatch.setattr(rel, "_domain_boost_terms", lambda: ("aws", "docker"))
    terms = rel.extract_key_terms("How does AWS work with foo.bar.0.name and Docker ingress?")
    assert {"aws", "docker", "ingress"} <= terms
    assert "foo.bar.0.name" not in terms


def test_overlap_empty_terms_is_one(monkeypatch):
    monkeypatch.setattr(rel, "extract_key_terms", lambda q: set())
    assert rel.term_overlap_score("the", article()) == 1.0


def test_overlap_scores_hits(monkeypatch):
    monkeypatch.setattr(rel, "extract_key_terms", lambda q: {"omega", "trial"})
    assert rel.term_overlap_score("q", article(body="omega trial")) == 1.0
    assert rel.term_overlap_score("q", article(body="omega")) == 0.5


def test_prefilter_noop_under_limit():
    items = [article(str(i)) for i in range(2)]
    kept, meta = rel.prefilter_articles_for_llm(items, "question", top_k=3)
    assert kept == items and meta["prefilter_applied"] is False


def test_prefilter_ranks_best(monkeypatch):
    monkeypatch.setattr(rel, "extract_key_terms", lambda q: {"omega", "trial"})
    weak = article("weak", "omega")
    strong = article("strong", "omega trial")
    kept, meta = rel.prefilter_articles_for_llm([weak, strong], "q", top_k=1)
    assert kept == [strong] and meta["prefilter_applied"] is True


def test_post_filter_keeps_matching(monkeypatch):
    monkeypatch.setattr(rel, "extract_key_terms", lambda q: {"omega", "trial"})
    monkeypatch.setattr("agent.faith_policy.post_filter_mode", lambda: "filter")
    monkeypatch.setattr("agent.tools.rerank.is_rerank_enabled", lambda: False)
    strong, weak = article("s", "omega trial"), article("w", "unrelated")
    kept, meta = rel.post_filter_relevant([weak, strong], "q", min_overlap=.5)
    assert kept == [strong] and meta["filtered_out"] == 1


def test_post_filter_rescues_top_k(monkeypatch):
    monkeypatch.setattr(rel, "extract_key_terms", lambda q: {"omega", "trial", "human"})
    monkeypatch.setattr("agent.faith_policy.post_filter_mode", lambda: "filter")
    monkeypatch.setattr("agent.tools.rerank.is_rerank_enabled", lambda: False)
    monkeypatch.setenv("AGENT_RESCUE_TOP_K", "1")
    best, bad = article("best", "omega"), article("bad", "none")
    kept, meta = rel.post_filter_relevant([bad, best], "q", min_overlap=.9)
    assert kept == [best] and meta["fallback_top_k"] is True


def test_dedupe_prefers_pmid():
    first = article("a", "one", pmid="123")
    duplicate = article("b", "two", pmid="123")
    assert rel.dedupe_articles([first, duplicate]) == [first]


def test_dedupe_by_normalized_body():
    first = article(body="Same   body")
    duplicate = article(body="same body")
    assert rel.dedupe_articles([first, duplicate]) == [first]
