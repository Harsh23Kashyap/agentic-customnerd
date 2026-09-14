from agent.state import AgentState
import agent.graph as g


def install(monkeypatch, assess=None, pre=None, post=None):
    monkeypatch.setattr(g,'tool_search_stackoverflow',lambda s,**k:(setattr(s,'search_attempts',s.search_attempts+1),setattr(s,'raw_articles',[{}])))
    monkeypatch.setattr(g,'tool_organize_articles',lambda s,**k:setattr(s,'organized_articles',[{}]))
    monkeypatch.setattr(g,'tool_classify_relevance',lambda s,**k:setattr(s,'relevant_articles',[{}]))
    monkeypatch.setattr(g,'tool_assess_retrieval',lambda *a,**k:assess or {'needs_research':False})
    calls={'verify':0}
    def verify(s,pre_synthesis=False,**k):
        calls['verify']+=1
        return (pre or {'passed':True}) if pre_synthesis else (post or {'passed':True})
    monkeypatch.setattr(g,'tool_verify_faithfulness',verify)
    monkeypatch.setattr(g,'tool_synthesize_answer',lambda s,**k:(setattr(s,'final_answer','answer'),setattr(s,'done',True)))
    return calls


def test_wrap():
    s=AgentState('s','q'); assert g._wrap(s)=={'agent_state':s,'phase':'start'}


def test_graph_happy(monkeypatch):
    calls=install(monkeypatch); s=AgentState('s','q'); s.max_search_attempts=2
    result=g.run_graph_agent(s)
    assert result.final_answer=='answer' and result.search_attempts==1 and calls['verify']==2


def test_graph_research_after_assess(monkeypatch):
    count={'n':0}
    def assessed(*a,**k):
        count['n']+=1; return {'needs_research':count['n']==1}
    install(monkeypatch); monkeypatch.setattr(g,'tool_assess_retrieval',assessed)
    s=AgentState('s','q'); s.max_search_attempts=2
    assert g.run_graph_agent(s).search_attempts==2


def test_graph_research_after_preverify(monkeypatch):
    count={'n':0}
    install(monkeypatch)
    def verify(s,pre_synthesis=False,**k):
        if pre_synthesis:
            count['n']+=1; return {'recommend_research':count['n']==1}
        return {'passed':True}
    monkeypatch.setattr(g,'tool_verify_faithfulness',verify)
    s=AgentState('s','q'); s.max_search_attempts=2
    assert g.run_graph_agent(s).search_attempts==2


def test_graph_regenerates_facts_only(monkeypatch):
    install(monkeypatch); count={'post':0,'synth':0}
    def verify(s,pre_synthesis=False,**k):
        if pre_synthesis:return {'passed':True}
        count['post']+=1; return {'recommend_regenerate':count['post']==1}
    def synth(s,**k): count['synth']+=1; s.final_answer='answer'; s.done=True
    monkeypatch.setattr(g,'tool_verify_faithfulness',verify); monkeypatch.setattr(g,'tool_synthesize_answer',synth)
    s=AgentState('s','q'); s.max_search_attempts=2
    result=g.run_graph_agent(s)
    assert count['synth']==2 and result.use_facts_only
