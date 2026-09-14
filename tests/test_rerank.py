import sys, types
import agent.tools.rerank as r


def test_settings(monkeypatch):
    monkeypatch.setenv('AGENT_RERANK','true'); monkeypatch.setenv('AGENT_RERANK_MODEL','m')
    assert r.is_rerank_enabled() and r.rerank_model_name()=='m'


def test_article_text():
    assert r._article_text({'title':'T','body':'B'})=='T B'


def test_score_falls_back(monkeypatch):
    monkeypatch.setattr(r,'_load_model',lambda:None)
    monkeypatch.setattr(r,'term_overlap_score',lambda q,a:.4)
    assert r.rerank_score('q',{})==.4


def test_score_cross_encoder(monkeypatch):
    class Model:
        def predict(self,pairs): return [0.9]
    monkeypatch.setattr(r,'_load_model',lambda:Model())
    assert r.rerank_score('q',{'body':'b'})==1.0


def test_rerank_disabled(monkeypatch):
    monkeypatch.setenv('AGENT_RERANK','false'); monkeypatch.setattr(r,'term_overlap_score',lambda q,a:a['s'])
    items=[{'s':1},{'s':2}]; assert r.rerank_articles('q',items,top_k=1)==[{'s':2}]


def test_rerank_enabled(monkeypatch):
    monkeypatch.setenv('AGENT_RERANK','true'); monkeypatch.setattr(r,'rerank_score',lambda q,a:a['s'])
    assert r.rerank_articles('q',[{'s':1},{'s':2}])==[{'s':2},{'s':1}]


def test_meta(monkeypatch):
    monkeypatch.setenv('AGENT_RERANK','true'); monkeypatch.setattr(r,'rerank_model_name',lambda:'m')
    assert r.rerank_meta('q',[{}])['enabled'] is True
