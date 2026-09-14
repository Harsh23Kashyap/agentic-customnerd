import sys, types
from agent.state import AgentState
import agent.tools.assess as assess


def install_legacy(monkeypatch, overlap=lambda q, f: .8):
    cascade=types.ModuleType('retrieval_cascade'); cascade.articles_to_source_counts=lambda a:{'strict':len(a)}; cascade.cascade_confidence=lambda m:'high'
    faith=types.ModuleType('faithfulness_two_pass'); faith.fact_term_overlap=overlap
    monkeypatch.setitem(sys.modules,'retrieval_cascade',cascade); monkeypatch.setitem(sys.modules,'faithfulness_two_pass',faith)
    monkeypatch.setattr(assess,'ensure_legacy_backend',lambda:None)


def st(raw=1, relevant=1):
    s=AgentState('s','omega trial'); s.raw_articles=[{'abstract':'omega trial'}]*raw; s.organized_articles=list(s.raw_articles); s.relevant_articles=list(s.raw_articles[:relevant]); s.search_attempts=1; s.max_search_attempts=2; s.cascade_meta={'retrieval_mode':'legacy'}; return s


def test_helpers(monkeypatch):
    monkeypatch.setenv('X','bad'); assert assess._env_int('X',3)==3
    assert assess._articles_with_answer_body([{'body':'x'},{}, {'abstract':'y'}])==2
    out=[]; assess._emit(out.append,'x'); assert out==['x']


def test_assess_zero_retrieval(monkeypatch):
    install_legacy(monkeypatch); s=st(0,0)
    result=assess.tool_assess_retrieval(s)
    assert result['needs_research'] and result['rescue_reason']=='zero_retrieval'


def test_assess_no_relevant(monkeypatch):
    install_legacy(monkeypatch); s=st(2,0)
    result=assess.tool_assess_retrieval(s)
    assert result['needs_research'] and result['rescue_reason']=='no_relevant'


def test_assess_good(monkeypatch):
    install_legacy(monkeypatch); s=st(2,2)
    result=assess.tool_assess_retrieval(s)
    assert result['confidence']=='high' and result['needs_research'] is False
    assert s.last_assessment is result


def test_verify_disabled(monkeypatch):
    monkeypatch.setenv('FAITHFULNESS_VERIFY_ENABLED','false'); s=st()
    assert assess.tool_verify_faithfulness(s)=={'passed':True,'skipped':True,'reason':'verify_disabled'}


def test_pre_verify_no_articles(monkeypatch):
    install_legacy(monkeypatch); s=st(0,0)
    assert assess.tool_verify_faithfulness(s,pre_synthesis=True)['reason']=='no_relevant_articles'


def test_pre_verify_success(monkeypatch):
    install_legacy(monkeypatch); s=st(); monkeypatch.setattr(assess,'get_or_extract_facts',lambda *a,**k:'facts'); monkeypatch.setattr(assess,'pre_synthesis_policy',lambda *a,**k:{'passed':True})
    assert assess.tool_verify_faithfulness(s,pre_synthesis=True)['passed']


def test_post_verify_empty(monkeypatch):
    install_legacy(monkeypatch); s=st(); s.final_answer=''
    assert assess.tool_verify_faithfulness(s)['reason']=='empty_answer'


def test_post_verify_policy(monkeypatch):
    install_legacy(monkeypatch); s=st(); s.final_answer='Omega helped [1].'; s.citations=['c']
    monkeypatch.setattr(assess,'get_or_extract_facts',lambda *a,**k:'facts')
    monkeypatch.setattr(assess,'post_synthesis_policy',lambda *a,**k:{'passed':True,**k})
    result=assess.tool_verify_faithfulness(s)
    assert result['passed'] and result['has_citations'] and result['ref_markers']==1
