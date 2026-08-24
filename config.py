"""Configuration for CloudNerd / DietNerd agentic RAG backend."""

from __future__ import annotations

import os
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AGENT_ROOT.parent
LEGACY_BACKEND_ROOT = REPO_ROOT / "customnerd-backend"

DEFAULT_PORT = int(os.getenv("PORT", "8001"))
MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4-turbo")
AGENT_MODE = os.getenv("AGENT_MODE", "hybrid").strip().lower()
# v6 default: 2 search attempts (was 3) to limit organize/classify repeats
MAX_SEARCH_ATTEMPTS = int(os.getenv("AGENT_MAX_SEARCH_ATTEMPTS", "2"))
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").strip().lower() in ("1", "true", "yes", "on")
HYBRID_RESCUE_ENABLED = os.getenv("HYBRID_RESCUE_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
# Soft hybrid rescue helps relevancy; set true for stricter cost
HYBRID_RESCUE_ON_ZERO_ONLY = os.getenv(
    "AGENT_HYBRID_RESCUE_ON_ZERO_ONLY", "false"
).strip().lower() in ("1", "true", "yes", "on")
AGENT_RUN_RETRIES = int(os.getenv("AGENT_RUN_RETRIES", "2"))
AGENT_FAITH_FIRST = os.getenv("AGENT_FAITH_FIRST", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
AGENT_POST_FILTER_MODE = os.getenv("AGENT_POST_FILTER_MODE", "").strip().lower()
# Domain pack under customnerd-backend/saved_states/ (CloudNerd | DietNerd | ...)
AGENT_DOMAIN = (os.getenv("AGENT_DOMAIN") or os.getenv("NERD_DOMAIN") or "CloudNerd").strip()


def load_environment() -> None:
    """Load env from legacy backend first, then local overrides, then domain pack."""
    from dotenv import load_dotenv

    legacy_env = LEGACY_BACKEND_ROOT / "variables.env"
    local_env = AGENT_ROOT / "variables.env"

    if legacy_env.is_file():
        load_dotenv(legacy_env, override=False)
    if local_env.is_file():
        load_dotenv(local_env, override=True)

    # Agent eval runs: deterministic synthesis (override legacy 0.2 default).
    os.environ.setdefault("FINAL_RESPONSE_TEMPERATURE", "0")
    # v6 cost defaults (override only if unset)
    os.environ.setdefault("AGENT_FAST_ORGANIZE", "true")
    os.environ.setdefault("AGENT_SKIP_URL_REENRICH", "true")
    os.environ.setdefault("AGENT_ORGANIZE_MAX_ARTICLES", "20")
    os.environ.setdefault("AGENT_RELEVANCE_PREFILTER_TOP_K", "10")
    # Soft hybrid rescue helps relevancy; zero-only was too aggressive for Rel.
    os.environ.setdefault("AGENT_HYBRID_RESCUE_ON_ZERO_ONLY", "false")
    # Balance faith + relevancy: grounded combined synthesis, query-focused
    # contexts, and thresholds that let grounded answers ship.
    os.environ.setdefault("AGENT_PREFER_COMBINED_SYNTHESIS", "true")
    os.environ.setdefault("AGENT_ORGANIZE_RELEVANT", "true")
    os.environ.setdefault("AGENT_FAITH_OVERLAP_THRESHOLD", "0.40")
    os.environ.setdefault("AGENT_MIN_EVIDENCE_OVERLAP_SHIP", "0.12")
    os.environ.setdefault("AGENT_MAX_SEARCH_ATTEMPTS", "2")
    os.environ.setdefault("AGENT_DOMAIN", AGENT_DOMAIN or "CloudNerd")

    # Activate domain pack (DietNerd → PubMed prompts/search; CloudNerd → SO).
    # Must run after dotenv so AGENT_DOMAIN from variables.env is visible.
    try:
        from bridge.domain import activate_domain

        activate_domain(os.getenv("AGENT_DOMAIN", "CloudNerd"), force=True)
    except Exception as exc:
        # Boot should still succeed; health endpoint will surface the warning.
        print(f"[config] domain activation warning: {exc}")
