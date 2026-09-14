import sys,types,json
from agent.state import AgentState
import agent.tools.claim_filter as c


def test_env_settings(monkeypatch):
    monkeypatch.setenv('AGENT_CLAIM_FILTER_MIN_CLAIMS','bad'); assert c.min_surviving_claims()==1
    monkeypatch.setenv('AGENT_CLAIM_FILTER_MIN_CLAIMS','0'); assert c.min_surviving_claims()==1


def test_empty_helpers():
    assert c.strip_disclaimer('')=='' and c.split_claims('')==[] and c._evidence_passages([])==''


def test_evidence_passages_max_chars():
    assert c._evidence_passages([{'abstract':'x'*100}],max_chars=10)==''


def client_module(content):
    msg=types.SimpleNamespace(content=content); response=types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)]); client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k:response)))
    m=types.ModuleType('openai_executions'); m.client=client; return m


def test_verify_claims_supported(monkeypatch):
    monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); monkeypatch.setitem(sys.modules,'openai_executions',client_module(json.dumps({'results':[{'claim':1,'verdict':'SUPPORTED'},{'claim':2,'verdict':'NOT SUPPORTED'}]})))
    assert c._verify_claims_batch(['one','two'],'passage')==['one']


def test_verify_claims_bad_json_keeps(monkeypatch):
    monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); monkeypatch.setitem(sys.modules,'openai_executions',client_module('bad'))
    assert c._verify_claims_batch(['one'],'passage')==['one']


def test_verify_no_client(monkeypatch):
    monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); m=types.ModuleType('openai_executions'); m.client=None; monkeypatch.setitem(sys.modules,'openai_executions',m)
    assert c._verify_claims_batch(['one'],'p')==['one']


def test_apply_empty_original():
    s=AgentState('s','q'); s.final_answer=''; out,meta=c.apply_claim_filter(s); assert out=='' and not meta['refuse']


def test_apply_single_supported(monkeypatch):
    s=AgentState('s','q'); s.final_answer='Claim.'; s.relevant_articles=[{'abstract':'Claim.'}]; monkeypatch.setattr(c,'_verify_claims_batch',lambda a,p:a)
    out,meta=c.apply_claim_filter(s); assert out=='Claim.' and meta['kept_claims']==1
