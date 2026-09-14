import json, types, sys
from agent.state import AgentState
import agent.orchestrator as o


def st(): return AgentState('s','question')


def test_limits_and_hybrid(monkeypatch):
    s=st(); o._init_state_limits(s); assert s.max_search_attempts==o.MAX_SEARCH_ATTEMPTS
    monkeypatch.setattr(o,'run_enhanced_pipeline',lambda s,**k:'ok'); assert o.run_hybrid_agent(s)=='ok'


def test_run_agent_modes(monkeypatch):
    monkeypatch.setattr(o,'AGENT_MODE','hybrid'); monkeypatch.setattr(o,'run_hybrid_agent',lambda s,**k:'h'); assert o.run_agent(st())=='h'
    monkeypatch.setattr(o,'AGENT_MODE','parity'); monkeypatch.setattr(o,'run_parity_pipeline',lambda s,**k:'p'); assert o.run_agent(st())=='p'
    monkeypatch.setattr(o,'AGENT_MODE','adaptive'); monkeypatch.setattr(o,'USE_LANGGRAPH',False); monkeypatch.setattr(o,'run_react_agent',lambda s,**k:'r'); assert o.run_agent(st())=='r'


def test_fallback_chain(monkeypatch):
    s=st(); calls=[]
    import agent.tools as t
    monkeypatch.setattr(t,'tool_search_stackoverflow',lambda s,**k:setattr(s,'raw_articles',[{}]) or calls.append('search'))
    monkeypatch.setattr(t,'tool_organize_articles',lambda s,**k:setattr(s,'organized_articles',[{}]) or calls.append('organize'))
    monkeypatch.setattr(t,'tool_classify_relevance',lambda s,**k:setattr(s,'relevant_articles',[{}]) or calls.append('classify'))
    monkeypatch.setattr(t,'tool_synthesize_answer',lambda s,**k:setattr(s,'done',True) or calls.append('synth'))
    monkeypatch.setattr(t,'tool_verify_faithfulness',lambda *a,**k:calls.append('verify'))
    monkeypatch.setattr(o,'tool_assess_retrieval',lambda *a,**k:calls.append('assess') or {})
    o._fallback_tool_chain(s,force_synthesize=True); assert calls==['search','organize','classify','assess','synth','verify']


def test_fallback_budget_message(monkeypatch):
    s=st(); s.raw_articles=[{}]; s.organized_articles=[{}]; s.relevant_articles=[]; s.search_attempts=0; s.max_search_attempts=2
    
    import agent.tools as t
    monkeypatch.setattr(t,'tool_classify_relevance',lambda s,**k:None)
    monkeypatch.setattr(o,'tool_assess_retrieval',lambda *a,**k:{})
    o._fallback_tool_chain(s); assert s.done and 'step budget' in s.final_answer


def test_build_result(monkeypatch):
    s=st(); s.final_answer='answer'; s.citations=['c']; s.raw_articles=[{}]; s.cascade_meta={'retrieval_mode':'legacy','winning_plane':2}; s.last_assessment={'confidence':'high'}
    monkeypatch.setattr(o,'_llm_call_snapshot',lambda:{'total':1}); monkeypatch.setattr(o,'_agent_domain',lambda:'DietNerd')
    result=o.build_result_object(s); assert result['end_output']=='answer' and result['confidence']=='high' and result['agent_domain']=='DietNerd'


def test_helpers(monkeypatch):
    monkeypatch.setenv('AGENT_DOMAIN','DietNerd'); assert o._agent_domain()=='DietNerd'
    assert isinstance(o._llm_call_snapshot(),dict)


def test_react_fallback_no_tool_calls(monkeypatch):
    s=st(); msg=types.SimpleNamespace(tool_calls=[],model_dump=lambda **k:{'role':'assistant'})
    resp=types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k:resp)))
    monkeypatch.setattr(o,'_client',lambda:client); monkeypatch.setattr(o,'MAX_AGENT_STEPS',1); monkeypatch.setattr(o,'_fallback_tool_chain',lambda s,**k:setattr(s,'done',True))
    assert o.run_react_agent(s).done


def test_react_tool_call(monkeypatch):
    s=st(); call=types.SimpleNamespace(id='1',function=types.SimpleNamespace(name='organize_articles',arguments='{}')); msg=types.SimpleNamespace(tool_calls=[call],model_dump=lambda **k:{'role':'assistant','tool_calls':[]}); resp=types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)]); client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k:resp)))
    monkeypatch.setattr(o,'_client',lambda:client); monkeypatch.setattr(o,'MAX_AGENT_STEPS',1); monkeypatch.setattr(o,'dispatch_tool',lambda *a,**k:'{"ok":true}'); monkeypatch.setattr(o,'_fallback_tool_chain',lambda s,**k:setattr(s,'done',True))
    assert o.run_react_agent(s).done
