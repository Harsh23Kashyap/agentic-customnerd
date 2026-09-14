import sys,types
from agent.state import AgentState
import agent.faith_policy as f


def s():
    x=AgentState('s','omega trial'); x.max_search_attempts=2; x.search_attempts=0; return x


def test_threshold_accessors(monkeypatch):
    monkeypatch.setenv('AGENT_FAITH_OVERLAP_THRESHOLD','.4'); monkeypatch.setenv('AGENT_FACTS_ONLY_HIGH_OVERLAP','.8'); monkeypatch.setenv('AGENT_MIN_EVIDENCE_OVERLAP_SHIP','.12')
    assert f.faith_overlap_threshold()==.4 and f.facts_only_bypass_overlap()==.8 and f.min_evidence_overlap_ship()==.12


def test_shallow_retrieval(monkeypatch):
    x=s(); x.cascade_meta={'winning_plane':3,'retrieval_mode':'strict'}; monkeypatch.setattr(f,'_has_fallback_sources',lambda s:False); assert not f.is_shallow_retrieval(x)
    x.cascade_meta['search_extension_used']=True; assert f.is_shallow_retrieval(x)


def test_context_top_k(monkeypatch):
    x=s(); x.domain_overlap=.5; monkeypatch.setenv('AGENT_ADAPTIVE_TOP_K','7'); assert f.effective_context_top_k(x)==7
    x.domain_overlap=.01; monkeypatch.setattr(f,'is_shallow_retrieval',lambda s:True); monkeypatch.setenv('AGENT_LOW_OVERLAP_TOP_K_EXTENDED','4'); assert f.effective_context_top_k(x)==4


def test_tighten_low_overlap(monkeypatch):
    x=s(); x.domain_overlap=.01; x.relevant_articles=[{'v':1},{'v':2},{'v':3}]
    monkeypatch.setattr(f,'effective_context_top_k',lambda s:2); monkeypatch.setattr('agent.tools.rerank.is_rerank_enabled',lambda:False); monkeypatch.setattr('agent.tools.relevance.term_overlap_score',lambda q,a:a['v'])
    assert f.tighten_relevant_articles(x)==[{'v':3},{'v':2}]
    assert x.suppress_gap_last and x.use_combined_lite


def test_get_facts_cached():
    x=s(); x.cached_evidence_facts='cached'; assert f.get_or_extract_facts(x)=='cached'


def test_get_facts_combined(monkeypatch):
    x=s(); x.relevant_articles=[{'abstract':'omega trial'}]
    monkeypatch.setattr(f,'prefer_combined_synthesis',lambda:True); monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None)
    mod=types.ModuleType('faithfulness_two_pass'); mod.fact_term_overlap=lambda q,facts:.5; monkeypatch.setitem(sys.modules,'faithfulness_two_pass',mod)
    assert f.get_or_extract_facts(x)=='omega trial' and x.cached_fact_overlap==.5


def test_get_facts_force(monkeypatch):
    x=s(); x.relevant_articles=[{}]; monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None)
    faith=types.ModuleType('faithfulness_two_pass'); faith.fact_term_overlap=lambda q,facts:.7; helper=types.ModuleType('helper_functions'); helper.extract_evidence_facts=lambda a,q,p:'facts'; prompts=types.ModuleType('openai_prompts'); prompts.EVIDENCE_EXTRACTION_PROMPT='p'
    for n,m in [('faithfulness_two_pass',faith),('helper_functions',helper),('openai_prompts',prompts)]:monkeypatch.setitem(sys.modules,n,m)
    assert f.get_or_extract_facts(x,force_refresh=True)=='facts'


def test_should_use_facts_branches(monkeypatch):
    x=s(); x.domain_overlap=.5; monkeypatch.setattr(f,'prefer_combined_synthesis',lambda:False); monkeypatch.setattr(f,'is_faith_first',lambda:True); x.relevant_articles=[{}]
    assert f.should_use_facts_only(x,force=True)
    assert f.should_use_facts_only(x,overlap=.1)
    assert not f.should_use_facts_only(x,overlap=.9)


def test_pre_policy_research(monkeypatch):
    x=s(); x.relevant_articles=[{}]; x.domain_overlap=.01; monkeypatch.setattr(f,'is_shallow_retrieval',lambda s:True); monkeypatch.setattr(f,'tighten_relevant_articles',lambda s:s.relevant_articles); monkeypatch.setattr(f,'should_use_facts_only',lambda *a,**k:False)
    out=f.pre_synthesis_policy(x,overlap=.1,threshold=.4)
    assert not out['passed'] and out['recommend_research'] and x.use_combined_lite


def test_post_policy_no_evidence():
    x=s(); out=f.post_synthesis_policy(x,overlap=1,answer_overlap=1,threshold=.4,ref_markers=0,has_citations=False,insufficient=False)
    assert out['reason']=='answer_without_relevant_evidence' and out['recommend_research']


def test_post_policy_weak():
    x=s(); x.relevant_articles=[{}]; out=f.post_synthesis_policy(x,overlap=0,answer_overlap=0,threshold=.4,ref_markers=1,has_citations=True,insufficient=False)
    assert out['reason']=='evidence_too_weak_to_ship'


def test_post_policy_missing_citations():
    x=s(); x.relevant_articles=[{}]; out=f.post_synthesis_policy(x,overlap=1,answer_overlap=1,threshold=.4,ref_markers=0,has_citations=False,insufficient=False)
    assert out['reason']=='missing_citation_markers' and out['recommend_regenerate']


def test_post_policy_low_overlap(monkeypatch):
    x=s(); x.relevant_articles=[{}]; x.search_attempts=2; monkeypatch.setattr(f,'is_shallow_retrieval',lambda s:False)
    out=f.post_synthesis_policy(x,overlap=.3,answer_overlap=.3,threshold=.4,ref_markers=1,has_citations=True,insufficient=False)
    assert out['reason']=='low_evidence_overlap' and out['recommend_regenerate']


def test_post_policy_answer_not_grounded(monkeypatch):
    x=s(); x.relevant_articles=[{}]; x.search_attempts=2; monkeypatch.setattr(f,'is_shallow_retrieval',lambda s:False)
    out=f.post_synthesis_policy(x,overlap=.8,answer_overlap=.2,threshold=.4,ref_markers=1,has_citations=True,insufficient=False)
    assert out['reason']=='answer_not_grounded_in_facts'


def test_ensure_citations(monkeypatch):
    x=s(); x.raw_articles=[{'title':'T','body':'B'}]; monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); helper=types.ModuleType('helper_functions'); helper.replace_invalid_values=lambda x:x; monkeypatch.setitem(sys.modules,'helper_functions',helper)
    f.ensure_citations_on_refusal(x); assert x.citations==['[1] T\nB']


def test_finalize_keeps_long_synthesis(monkeypatch):
    x=s(); x.final_answer='A'*130; x.relevant_articles=[{}]; x.domain_overlap=.5; monkeypatch.setattr(f,'ensure_citations_on_refusal',lambda s:None)
    assert f.finalize_or_refuse_answer(x,{'passed':False,'reason':'evidence_too_weak_to_ship','evidence_overlap':.2,'answer_fact_overlap':.2},overlap_rescue_attempted=True)
    assert x.final_answer=='A'*130 and x.agent_trace[-1]['detail']['shipped']=='synthesized_partial'
