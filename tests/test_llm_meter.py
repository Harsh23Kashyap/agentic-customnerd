from agent import llm_meter


def test_meter_reset_increment_snapshot():
    llm_meter.reset()
    llm_meter.incr_chat("gpt", 2)
    llm_meter.incr_embeddings("embed", 3)
    snap = llm_meter.snapshot()
    assert snap == {"chat": 2, "embeddings": 3, "total": 5, "by_model": {"gpt": 2}}


def test_install_idempotent():
    llm_meter.install()
    llm_meter.install()
    assert llm_meter._installed is True

def test_increment_defaults_and_copy():
    llm_meter.reset(); llm_meter.incr_chat(); llm_meter.incr_embeddings()
    snap=llm_meter.snapshot(); snap['by_model']['mutated']=1
    assert llm_meter.snapshot()=={'chat':1,'embeddings':1,'total':2,'by_model':{'unknown':1}}
