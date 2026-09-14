from pathlib import Path
import bridge.domain as domain


def test_domain_aliases(monkeypatch):
    monkeypatch.setenv("AGENT_DOMAIN", "diet-nerd")
    assert domain.is_dietnerd() is True and domain.is_cloudnerd() is False


def test_list_domains(tmp_path, monkeypatch):
    root = tmp_path / "saved_states"
    (root / "DietNerd").mkdir(parents=True)
    (root / "CloudNerd").mkdir()
    monkeypatch.setattr(domain, "LEGACY_BACKEND_ROOT", tmp_path)
    assert domain.list_domains() == ["CloudNerd", "DietNerd"]


def test_copy_domain_files_only_known_files(tmp_path, monkeypatch):
    state = tmp_path / "saved_states" / "DietNerd"
    state.mkdir(parents=True)
    (state / "openai_prompts.py").write_text("PROMPT=1")
    (state / "variables.env").write_text("SECRET=do-not-copy")
    monkeypatch.setattr(domain, "LEGACY_BACKEND_ROOT", tmp_path)
    copied = domain._copy_domain_files("DietNerd")
    assert copied == ["openai_prompts.py"]
    assert (tmp_path / "openai_prompts.py").read_text() == "PROMPT=1"
    assert not (tmp_path / "variables.env").exists()


def test_missing_domain_reports_available(tmp_path, monkeypatch):
    (tmp_path / "saved_states" / "CloudNerd").mkdir(parents=True)
    monkeypatch.setattr(domain, "LEGACY_BACKEND_ROOT", tmp_path)
    try:
        domain._copy_domain_files("Missing")
    except FileNotFoundError as exc:
        assert "CloudNerd" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_diet_defaults(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    domain.apply_domain_defaults("DietNerd")
    assert domain.get_domain() == "DietNerd"
    assert domain.is_dietnerd() is True
    assert __import__("os").environ["RETRIEVAL_MODE"] == "legacy"

def test_saved_state_dir(monkeypatch,tmp_path):
    monkeypatch.setattr(domain,'LEGACY_BACKEND_ROOT',tmp_path)
    assert domain.saved_state_dir('DietNerd')==tmp_path/'saved_states'/'DietNerd'


def test_purge_modules(monkeypatch):
    import sys,types
    sys.modules['_temp_domain_module']=types.ModuleType('_temp_domain_module')
    domain._purge_modules(['_temp_domain_module','missing'])
    assert '_temp_domain_module' not in sys.modules


def test_cloud_defaults(monkeypatch):
    monkeypatch.delenv('RETRIEVAL_MODE',raising=False); domain.apply_domain_defaults('CloudNerd')
    import os
    assert os.environ['RETRIEVAL_MODE']=='cascade'
