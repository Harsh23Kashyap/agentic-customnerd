import sys, types
from agent.state import AgentState
import agent.tools.organize as org


def test_env_and_type_helpers(monkeypatch):
    monkeypatch.setenv('B','off'); monkeypatch.setenv('I','bad')
    assert org._env_bool('B') is False and org._env_int('I',4)==4
    assert org._looks_like_pubmed({'MedlineCitation':{}})
    assert org._looks_like_so({'answer_body':'x','answer_id':1})
    assert org._entrez_text([{'#text':'a'},'b'])=='a b'
    assert org._strip_html('<p>Hello</p>')=='Hello'
    assert org._truncate('one two three',7)=='one...'


def test_fast_pubmed_mapping():
    raw={'MedlineCitation':{'PMID':'123','Article':{'ArticleTitle':'Title','Abstract':{'AbstractText':['Abstract text']},'Journal':{'Title':'J','JournalIssue':{'PubDate':{'Year':'2024'}}},'AuthorList':[{'LastName':'Doe','Initials':'J'}]}},'retrieval_source':'pubmed'}
    out=org._fast_organize_pubmed_article(raw)
    assert out['PMID']=='123' and out['url'].endswith('/123/') and out['author_name']=='Doe J'


def test_fast_so_mapping():
    out=org._fast_organize_so_article({'question_title':'T','answer_body':'<p>Use it</p>','answer_id':4,'score':2})
    assert out['title']=='T' and out['id']=='4' and '<p>' not in out['answer_body']


def test_cap_raw(monkeypatch):
    monkeypatch.setattr(org,'organize_max_articles',lambda:1)
    monkeypatch.setattr('agent.tools.relevance.term_overlap_score',lambda q,a:a['score'])
    assert org.cap_raw_articles_for_organize([{'score':1},{'score':2}],'q')==[{'score':2}]


def legacy(monkeypatch):
    helper=types.ModuleType('helper_functions'); helper.concurrent_organize_database_articles=lambda a,q:[{'title':'slow'} for _ in a]; helper.process_articles_by_url=lambda a:a
    monkeypatch.setitem(sys.modules,'helper_functions',helper); monkeypatch.setattr(org,'ensure_legacy_backend',lambda:None)


def test_tool_no_articles(monkeypatch):
    legacy(monkeypatch); s=AgentState('s','q')
    assert org.tool_organize_articles(s)['organized_count']==0


def test_tool_fast_mixed(monkeypatch):
    legacy(monkeypatch); monkeypatch.setenv('AGENT_FAST_ORGANIZE','true'); monkeypatch.setenv('AGENT_SKIP_URL_REENRICH','true')
    s=AgentState('s','q'); s.raw_articles=[{'question_title':'T','answer_body':'body','answer_id':1},{'weird':1}]
    result=org.tool_organize_articles(s)
    assert result['organized_count']==2 and result['llm_organize']==1


def test_tool_slow(monkeypatch):
    legacy(monkeypatch); monkeypatch.setenv('AGENT_FAST_ORGANIZE','false')
    s=AgentState('s','q'); s.raw_articles=[{'x':1}]
    assert org.tool_organize_articles(s)['fast_organize'] is False


def test_summarize_disabled(monkeypatch):
    monkeypatch.setattr(org,'is_organize_relevant_enabled',lambda:False)
    items=[{'body':'x'}]; assert org.summarize_relevant_articles(items,'q') is items

def test_relevant_summary_prompt_diet(monkeypatch):
    mod=types.ModuleType('bridge.domain'); mod.is_dietnerd=lambda:True; monkeypatch.setitem(sys.modules,'bridge.domain',mod)
    assert 'PubMed' in org._relevant_summary_prompt()


def test_summarize_one_short_returns_same():
    item={'body':'short'}; assert org._summarize_one(item,'q') is item


def test_summarize_one_with_client(monkeypatch):
    item={'answer_body':'x'*60}
    msg=types.SimpleNamespace(content='focused summary'); resp=types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)]); client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k:resp))); mod=types.ModuleType('openai_executions'); mod.client=client; monkeypatch.setitem(sys.modules,'openai_executions',mod)
    out=org._summarize_one(item,'q'); assert out['summary']=='focused summary' and len(out['abstract'])==60


def test_summarize_one_no_client(monkeypatch):
    item={'answer_body':'x'*60}; mod=types.ModuleType('openai_executions'); mod.client=None; monkeypatch.setitem(sys.modules,'openai_executions',mod)
    assert org._summarize_one(item,'q') is item


def test_summarize_parallel(monkeypatch):
    monkeypatch.setattr(org,'is_organize_relevant_enabled',lambda:True); monkeypatch.setattr(org,'_summarize_one',lambda a,q:{**a,'summary':'s'})
    assert org.summarize_relevant_articles([{'body':'a'},{'body':'b'}],'q')==[{'body':'a','summary':'s'},{'body':'b','summary':'s'}]
