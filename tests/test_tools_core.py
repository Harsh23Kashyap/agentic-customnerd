import sys, types, json
from agent.state import AgentState
import agent.tools as tools


def state():
    s=AgentState('s','omega question'); s.max_search_attempts=2; return s


def modules(monkeypatch, diet=True):
    monkeypatch.setattr(tools,'ensure_legacy_backend',lambda:None)
    monkeypatch.setattr(tools,'cascade_body_for_tier',lambda s,q,t:(q,'title'))
    dom=types.ModuleType('bridge.domain'); dom.is_dietnerd=lambda:diet
    helper=types.ModuleType('helper_functions'); helper.query_generation=lambda q:(None,None,[q]); helper.concurrent_relevance_classification=lambda a,q:(a,[]); helper.trim_relevant_articles_by_token_limit=lambda a,q:a; helper.generate_final_response=lambda a,q,**kw:'A grounded answer from evidence with enough detail to be useful and direct [1].'; helper.print_referenced_articles=lambda answer,a:['citation']; helper.replace_invalid_values=lambda x:x
    monkeypatch.setitem(sys.modules,'bridge.domain',dom); monkeypatch.setitem(sys.modules,'helper_functions',helper)


def test_search_diet_new_and_duplicate(monkeypatch):
    modules(monkeypatch,True); monkeypatch.setattr(tools,'collect_with_fallback',lambda *a,**k:[{'PMID':'1','abstract':'omega'}])
    s=state(); first=tools.tool_search_stackoverflow(s)
    assert first['corpus']=='PubMed' and first['new_unique']==1
    s.organized_articles=[{'PMID':'1'}]
    second=tools.tool_search_stackoverflow(s)
    assert second['skipped_reprocess'] is True and second['new_unique']==0


def test_search_empty(monkeypatch):
    modules(monkeypatch,True); monkeypatch.setattr(tools,'collect_with_fallback',lambda *a,**k:[])
    assert tools.tool_search_stackoverflow(state())['total_articles']==0


def test_classify_empty(monkeypatch):
    modules(monkeypatch); s=state(); assert tools.tool_classify_relevance(s)['relevant_count']==0


def test_classify_normal(monkeypatch):
    modules(monkeypatch); s=state(); s.raw_articles=[{'abstract':'omega question'}]
    monkeypatch.setattr(tools,'prefilter_articles_for_llm',lambda a,q:(a,{'prefilter_top_k':10,'prefilter_applied':False}))
    monkeypatch.setattr(tools,'post_filter_relevant',lambda a,q:(a,{'filtered_out':0}))
    monkeypatch.setattr(tools,'cap_relevant_articles',lambda a,q:a)
    result=tools.tool_classify_relevance(s)
    assert result['relevant_count']==1 and s.relevant_articles


def test_classify_floor(monkeypatch):
    modules(monkeypatch); s=state(); s.raw_articles=[{'abstract':'omega'}]
    monkeypatch.setattr(tools,'prefilter_articles_for_llm',lambda a,q:(a,{'prefilter_top_k':10}))
    helper=sys.modules['helper_functions']; helper.concurrent_relevance_classification=lambda a,q:([],a)
    monkeypatch.setattr(tools,'post_filter_relevant',lambda a,q:([],{})); monkeypatch.setattr(tools,'cap_relevant_articles',lambda a,q:a)
    rerank=types.ModuleType('agent.tools.rerank'); rerank.is_rerank_enabled=lambda:False; rerank.rerank_articles=lambda *a,**k:[]
    rank=types.ModuleType('retrieval_rank'); rank.rank_articles=lambda q,a,top_k:a[:top_k]
    monkeypatch.setitem(sys.modules,'agent.tools.rerank',rerank); monkeypatch.setitem(sys.modules,'retrieval_rank',rank)
    assert tools.tool_classify_relevance(s)['relevant_count']==1


def test_synthesize_skips(monkeypatch):
    modules(monkeypatch); s=state(); assert tools.tool_synthesize_answer(s)['skipped']


def test_synthesize_empty_after_trim(monkeypatch):
    modules(monkeypatch); s=state(); s.relevant_articles=[{'abstract':'omega'}]; s.search_attempts=2
    sys.modules['helper_functions'].trim_relevant_articles_by_token_limit=lambda a,q:[]
    monkeypatch.setattr(tools,'prepare_synthesis_context',lambda s:None); monkeypatch.setattr(tools,'cap_relevant_articles',lambda a,q:a)
    faith=types.ModuleType('agent.faith_policy'); faith.build_grounded_fallback_answer=lambda s:'fallback'; faith.ensure_citations_on_refusal=lambda s:None
    monkeypatch.setitem(sys.modules,'agent.faith_policy',faith)
    assert tools.tool_synthesize_answer(s)['answer_length']==8


def test_synthesize_combined(monkeypatch):
    modules(monkeypatch); s=state(); s.relevant_articles=[{'abstract':'omega evidence','_qsummarized':True}]; s.search_attempts=2; s.use_combined_lite=True
    monkeypatch.setattr(tools,'prepare_synthesis_context',lambda s:None); monkeypatch.setattr(tools,'cap_relevant_articles',lambda a,q:a); monkeypatch.setattr(tools,'should_use_facts_only',lambda *a,**k:False)
    faith=types.ModuleType('faithfulness_two_pass'); faith.is_two_pass_enabled=lambda:False
    monkeypatch.setitem(sys.modules,'faithfulness_two_pass',faith)
    result=tools.tool_synthesize_answer(s)
    assert result['answer_length']>20 and s.done and s.citations==['citation']


def test_synthesize_facts(monkeypatch):
    modules(monkeypatch); s=state(); s.relevant_articles=[{'abstract':'omega evidence','_qsummarized':True}]; s.search_attempts=2
    monkeypatch.setattr(tools,'prepare_synthesis_context',lambda s:None); monkeypatch.setattr(tools,'cap_relevant_articles',lambda a,q:a); monkeypatch.setattr(tools,'should_use_facts_only',lambda *a,**k:True); monkeypatch.setattr(tools,'get_or_extract_facts',lambda *a,**k:'facts')
    faith=types.ModuleType('faithfulness_two_pass'); faith.is_two_pass_enabled=lambda:True
    monkeypatch.setitem(sys.modules,'faithfulness_two_pass',faith)
    assert tools.tool_synthesize_answer(s,force_facts_only=True)['facts_only']


def test_dispatch_all(monkeypatch):
    s=state(); names=['search_stackoverflow','organize_articles','classify_relevance','assess_retrieval','synthesize_answer','verify_faithfulness','reflect_on_answer']
    monkeypatch.setattr(tools,'tool_search_stackoverflow',lambda *a,**k:{'x':1}); monkeypatch.setattr(tools,'tool_organize_articles',lambda *a,**k:{'x':1}); monkeypatch.setattr(tools,'tool_classify_relevance',lambda *a,**k:{'x':1}); monkeypatch.setattr(tools,'tool_assess_retrieval',lambda *a,**k:{'x':1}); monkeypatch.setattr(tools,'tool_synthesize_answer',lambda *a,**k:{'x':1}); monkeypatch.setattr(tools,'tool_verify_faithfulness',lambda *a,**k:{'x':1})
    for n in names: assert json.loads(tools.dispatch_tool(n,{},s))['x']==1
    assert 'error' in json.loads(tools.dispatch_tool('bad',{},s))
