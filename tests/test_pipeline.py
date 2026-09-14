from agent.state import AgentState
import agent.pipeline as p


def s(): return AgentState('s','question')


def test_helpers(monkeypatch):
    monkeypatch.setenv('AGENT_HYBRID_RESCUE_ON_ZERO_ONLY','yes'); assert p._hybrid_rescue_on_zero_only()
    assert p._is_shippable_synthesis('A detailed answer that directly explains the relevant configuration and provides enough useful context.')
    assert not p._is_shippable_synthesis('Answer for: raw')


def test_search_classify_assess(monkeypatch):
    x=s(); calls=[]
    monkeypatch.setattr(p,'tool_search_stackoverflow',lambda *a,**k:calls.append('search') or {'skipped_reprocess':False})
    monkeypatch.setattr(p,'tool_organize_articles',lambda *a,**k:calls.append('organize'))
    monkeypatch.setattr(p,'tool_classify_relevance',lambda *a,**k:calls.append('classify'))
    monkeypatch.setattr(p,'tool_assess_retrieval',lambda *a,**k:calls.append('assess') or {'ok':1})
    assert p._search_classify_assess(x)=={'ok':1}; assert calls==['search','organize','classify','assess']


def test_ship_synthesis(monkeypatch):
    x=s(); x.final_answer='A detailed answer that directly explains the relevant configuration and provides enough useful context.'
    monkeypatch.setattr(p,'ensure_citations_on_refusal',lambda s:s.citations.append('c'))
    p._ship_grounded_fallback(x,step='partial')
    assert x.done and x.citations==['c'] and x.agent_trace[-1]['detail']['shipped']=='synthesized_partial'


def test_ship_collage(monkeypatch):
    x=s(); x.final_answer='bad'
    monkeypatch.setattr(p,'build_grounded_fallback_answer',lambda s:'grounded [1]')
    monkeypatch.setattr(p,'ensure_citations_on_refusal',lambda s:None)
    p._ship_grounded_fallback(x,step='partial')
    assert x.final_answer=='grounded [1]'


def test_apply_filter_noop(monkeypatch):
    x=s(); x.final_answer='answer'
    monkeypatch.setattr(p,'is_claim_filter_enabled',lambda:False)
    assert p._apply_claim_filter_step(x) is False


def test_apply_filter_changes(monkeypatch):
    x=s(); x.final_answer='one. two.'
    monkeypatch.setattr(p,'is_claim_filter_enabled',lambda:True); monkeypatch.setattr(p,'apply_claim_filter',lambda s:('one.',{'kept_claims':1}))
    assert p._apply_claim_filter_step(x) is False and x.final_answer=='one.'


def test_apply_filter_refuses(monkeypatch):
    x=s(); x.final_answer='long answer'
    monkeypatch.setattr(p,'is_claim_filter_enabled',lambda:True); monkeypatch.setattr(p,'apply_claim_filter',lambda s:('',{'refuse':True}))
    monkeypatch.setattr(p,'_ship_grounded_fallback',lambda s,**k:setattr(s,'done',True))
    assert p._apply_claim_filter_step(x) is True and x.done


def test_overlap_rescue_success(monkeypatch):
    x=s(); monkeypatch.setattr(p,'tool_synthesize_answer',lambda s,**k:setattr(s,'final_answer','rescued answer') or {})
    monkeypatch.setattr(p,'apply_claim_filter',lambda s:('rescued answer',{'kept_claims':1,'refuse':False}))
    assert p._attempt_overlap_rescue(x) and x.done


def test_maybe_refuse_passed():
    assert p._maybe_refuse_with_rescue(s(),{'passed':True}) is False


def setup_flow(monkeypatch, post=None, assess=None):
    monkeypatch.setattr(p,'_search_classify_assess',lambda *a,**k:assess or {'needs_research':False,'needs_rescue':False})
    monkeypatch.setattr(p,'tool_verify_faithfulness',lambda state,pre_synthesis=False,**k: ({'passed':True} if pre_synthesis else (post or {'passed':True,'answer_fact_overlap':1})))
    monkeypatch.setattr(p,'tool_synthesize_answer',lambda state,**k:setattr(state,'final_answer','This is a sufficiently detailed grounded answer with citations and a concrete useful recommendation [1].') or {})
    monkeypatch.setattr(p,'_apply_claim_filter_step',lambda *a,**k:False)


def test_enhanced_happy(monkeypatch):
    setup_flow(monkeypatch); x=p.run_enhanced_pipeline(s()); assert x.final_answer and x.done is False


def test_enhanced_research(monkeypatch):
    setup_flow(monkeypatch,assess={'needs_research':True,'rescue_reason':'no_relevant','needs_rescue':False})
    monkeypatch.setattr(p,'_research_with_focus',lambda *a,**k:{'needs_research':False,'needs_rescue':False})
    x=s(); x.max_search_attempts=2; x.search_attempts=0
    assert p.run_enhanced_pipeline(x).final_answer


def test_parity_delegates(monkeypatch):
    monkeypatch.setattr(p,'run_enhanced_pipeline',lambda s,**k:'ok'); assert p.run_parity_pipeline(s())=='ok'

def test_research_with_focus(monkeypatch):
    x=s(); x.search_attempts=1; seen={}
    monkeypatch.setattr(p,'_search_classify_assess',lambda state,**kw:seen.update(kw) or {'ok':1})
    monkeypatch.setattr(p,'build_search_focus',lambda *a,**k:'focus')
    assert p._research_with_focus(x,reason='low')=={'ok':1} and seen['search_focus']=='focus'


def test_attempt_overlap_rescue_skipped(monkeypatch):
    x=s(); monkeypatch.setattr(p,'tool_synthesize_answer',lambda *a,**k:{'skipped':True})
    assert not p._attempt_overlap_rescue(x)


def test_maybe_refuse_finalizes(monkeypatch):
    x=s(); monkeypatch.setattr(p,'_attempt_overlap_rescue',lambda *a,**k:False); monkeypatch.setattr(p,'finalize_or_refuse_answer',lambda *a,**k:True)
    assert p._maybe_refuse_with_rescue(x,{'passed':False})


def test_enhanced_claim_filter_stops(monkeypatch):
    setup_flow(monkeypatch); monkeypatch.setattr(p,'_apply_claim_filter_step',lambda *a,**k:True)
    x=s(); assert p.run_enhanced_pipeline(x) is x


def test_enhanced_synth_skip_research(monkeypatch):
    monkeypatch.setattr(p,'_search_classify_assess',lambda *a,**k:{'needs_research':False,'needs_rescue':False})
    monkeypatch.setattr(p,'tool_verify_faithfulness',lambda state,pre_synthesis=False,**k:{'passed':True,'answer_fact_overlap':1})
    count={'n':0}
    def synth(state,**k): count['n']+=1; state.final_answer='This is a sufficiently detailed answer that contains grounded evidence and citations [1].'; return {'skipped':count['n']==1}
    monkeypatch.setattr(p,'tool_synthesize_answer',synth); monkeypatch.setattr(p,'_research_with_focus',lambda *a,**k:{'needs_rescue':False}); monkeypatch.setattr(p,'_apply_claim_filter_step',lambda *a,**k:False)
    x=s(); x.max_search_attempts=2; assert p.run_enhanced_pipeline(x).final_answer and count['n']==2

def test_is_shippable_edge_cases():
    assert not p._is_shippable_synthesis('short')
    assert not p._is_shippable_synthesis('I could not find enough evidence to answer this detailed user question with confidence.')


def test_apply_filter_sets_filtered_when_equal(monkeypatch):
    x=s(); x.final_answer='same answer.'; monkeypatch.setattr(p,'is_claim_filter_enabled',lambda:True); monkeypatch.setattr(p,'apply_claim_filter',lambda s:('same answer.',{'kept_claims':1}))
    assert p._apply_claim_filter_step(x) is False and x.final_answer=='same answer.'
