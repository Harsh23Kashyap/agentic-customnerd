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
