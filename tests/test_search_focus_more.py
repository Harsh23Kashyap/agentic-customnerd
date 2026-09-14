import sys,types
from agent.state import AgentState
import agent.tools.search_focus as sf


def st(): return AgentState('s','AWS docker kubernetes timeout',query_title='A useful title')


def test_truncated_title_short(monkeypatch):
    monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); mod=types.ModuleType('retrieval_cascade'); mod.resolve_title=lambda t,q:t; monkeypatch.setitem(sys.modules,'retrieval_cascade',mod)
    assert sf.truncated_title_for_similar(st())=='A useful title'


def test_truncated_title_long(monkeypatch):
    monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); mod=types.ModuleType('retrieval_cascade'); mod.resolve_title=lambda t,q:'First line?'+('x'*200); monkeypatch.setitem(sys.modules,'retrieval_cascade',mod)
    assert sf.truncated_title_for_similar(st(),max_len=20)=='First line'


def test_hint_nondeterministic_zero(monkeypatch):
    monkeypatch.setenv('AGENT_DETERMINISTIC_LADDER','false'); x=st(); x.raw_articles=[]
    assert sf.suggest_tier_hint(x,reason='zero_retrieval')=='extended_keywords'


def test_hint_nondeterministic_no_relevant(monkeypatch):
    monkeypatch.setenv('AGENT_DETERMINISTIC_LADDER','false'); x=st(); x.raw_articles=[{}]; x.cascade_meta={'winning_plane':3,'retrieval_mode':'strict'}
    assert sf.suggest_tier_hint(x,reason='no_relevant')=='title-only'


def install(monkeypatch):
    monkeypatch.setattr('bridge.legacy.ensure_legacy_backend',lambda:None); mod=types.ModuleType('retrieval_cascade'); mod.resolve_title=lambda t,q:t or q; monkeypatch.setitem(sys.modules,'retrieval_cascade',mod)


def test_cascade_bodies(monkeypatch):
    install(monkeypatch); monkeypatch.setattr(sf,'extract_key_terms',lambda q:{'docker','timeout'})
    x=st();
    body,title=sf.cascade_body_for_tier(x,'query','title-only'); assert body=='A useful title'
    body,title=sf.cascade_body_for_tier(x,'query','extended_keywords'); assert 'docker' in body
    monkeypatch.setattr(sf,'truncated_title_for_similar',lambda s:'short'); body,title=sf.cascade_body_for_tier(x,'query','tag_similar'); assert title=='short'
    body,title=sf.cascade_body_for_tier(x,'query','strict'); assert body=='query'
