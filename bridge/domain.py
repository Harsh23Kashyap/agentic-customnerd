"""Domain activation: load a saved Nerd state into the live legacy backend.

DietNerd / CloudNerd / etc. are file packs under
`customnerd-backend/saved_states/<name>/`. Activating a domain copies those
files into the live backend root and clears cached imports so the agent
picks up PubMed (Diet) vs Stack Overflow (Cloud) retrieval + prompts.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

from config import LEGACY_BACKEND_ROOT

logger = logging.getLogger("cloudnerd-agent.domain")

# Modules that come from the domain pack and must be reloaded after a switch.
_DOMAIN_MODULES = (
    "openai_prompts",
    "user_search_apis",
    "user_list_search",
    "clean_query",
)

_BACKEND_FILES = (
    "openai_prompts.py",
    "user_search_apis.py",
    "user_list_search.py",
    "clean_query.py",
    # Intentionally NOT copying variables.env — API keys are shared across domains.
)


def get_domain() -> str:
    return (os.getenv("AGENT_DOMAIN") or os.getenv("NERD_DOMAIN") or "CloudNerd").strip()


def is_dietnerd() -> bool:
    return get_domain().lower() in {"dietnerd", "diet", "diet-nerd"}


def is_cloudnerd() -> bool:
    return not is_dietnerd()


def saved_state_dir(domain: Optional[str] = None) -> Path:
    name = (domain or get_domain()).strip()
    return LEGACY_BACKEND_ROOT / "saved_states" / name


def list_domains() -> list[str]:
    root = LEGACY_BACKEND_ROOT / "saved_states"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def _purge_modules(names: Iterable[str]) -> None:
    for name in names:
        if name in sys.modules:
            del sys.modules[name]


def _copy_domain_files(domain: str) -> list[str]:
    state_dir = saved_state_dir(domain)
    if not state_dir.is_dir():
        raise FileNotFoundError(
            f"Domain '{domain}' not found at {state_dir}. "
            f"Available: {list_domains()}"
        )

    copied: list[str] = []
    for fname in _BACKEND_FILES:
        src = state_dir / fname
        if not src.is_file():
            continue
        dest = LEGACY_BACKEND_ROOT / fname
        shutil.copy2(src, dest)
        copied.append(fname)
    return copied


def apply_domain_defaults(domain: Optional[str] = None) -> None:
    """Set agent env defaults that differ between CloudNerd and DietNerd.

    Uses setdefault so explicit variables.env / process env still win.
    """
    name = (domain or get_domain()).strip()
    os.environ["AGENT_DOMAIN"] = name

    if name.lower() in {"dietnerd", "diet", "diet-nerd"}:
        # PubMed Entrez path — SO cascade does not apply; fast Entrez→article map does.
        os.environ["RETRIEVAL_MODE"] = "legacy"
        os.environ.setdefault("AGENT_FAST_ORGANIZE", "true")
        os.environ.setdefault("AGENT_SKIP_URL_REENRICH", "true")
        # Combined-lite still works once Diet prompts include it.
        os.environ.setdefault("AGENT_PREFER_COMBINED_SYNTHESIS", "true")
        os.environ.setdefault("AGENT_ORGANIZE_RELEVANT", "true")
        os.environ.setdefault("AGENT_CLAIM_FILTER", "true")
        # Soften SO-specific hybrid rescue defaults are fine to keep.
    else:
        os.environ.setdefault("RETRIEVAL_MODE", "cascade")
        os.environ.setdefault("AGENT_FAST_ORGANIZE", "true")


def activate_domain(domain: Optional[str] = None, *, force: bool = True) -> dict:
    """Copy saved-state files into live backend and reload domain modules.

    Returns a status dict for health / logging.
    """
    name = (domain or get_domain()).strip()
    if not name:
        name = "CloudNerd"
    os.environ["AGENT_DOMAIN"] = name

    copied: list[str] = []
    if force:
        copied = _copy_domain_files(name)
        _purge_modules(_DOMAIN_MODULES)

    apply_domain_defaults(name)

    # Touch imports so failures surface at boot, not mid-request.
    from bridge.legacy import ensure_legacy_backend

    ensure_legacy_backend()
    import openai_prompts  # noqa: F401
    import user_search_apis  # noqa: F401
    import clean_query  # noqa: F401

    status = {
        "domain": name,
        "state_dir": str(saved_state_dir(name)),
        "copied_files": copied,
        "retrieval_mode": os.getenv("RETRIEVAL_MODE"),
        "fast_organize": os.getenv("AGENT_FAST_ORGANIZE"),
        "is_dietnerd": is_dietnerd(),
    }
    logger.info("Activated domain %s (copied=%s)", name, copied)
    return status
