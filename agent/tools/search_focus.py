"""Structured search-focus query builder for agent re-search."""

from __future__ import annotations

import os
import re
from typing import Optional

from agent.state import AgentState
from agent.tools.relevance import extract_key_terms

_SEARCH_LADDER = ("strict", "extended_keywords", "tag_similar")
_LONG_BODY_LADDER = ("tag_similar", "extended_keywords", "strict")
_LONG_BODY_CHARS = 1500


def _detect_providers(query: str) -> list[str]:
    lower = (query or "").lower()
    found: list[str] = []
    _PROVIDER_TERMS = {
        "aws": "AWS",
        "amazon": "AWS",
        "azure": "Azure",
        "gcp": "Google Cloud",
        "google cloud": "Google Cloud",
        "kubernetes": "Kubernetes",
        "k8s": "Kubernetes",
        "terraform": "Terraform",
        "docker": "Docker",
        "heroku": "Heroku",
        "flutter": "Flutter",
        "dart": "Flutter",
    }
    for key, label in _PROVIDER_TERMS.items():
        if key in lower and label not in found:
            found.append(label)
    return found


def is_deterministic_ladder() -> bool:
    raw = os.getenv("AGENT_DETERMINISTIC_LADDER", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _body_length(state: AgentState | None) -> int:
    if not state:
        return 0
    return len((state.user_query or "").strip())


def _search_ladder(state: AgentState | None) -> tuple[str, ...]:
    if _body_length(state) > _LONG_BODY_CHARS:
        return _LONG_BODY_LADDER
    return _SEARCH_LADDER


def tier_for_search_attempt(attempt: int, *, state: AgentState | None = None) -> str:
    ladder = _search_ladder(state)
    idx = max(0, min(int(attempt or 1) - 1, len(ladder) - 1))
    return ladder[idx]


def truncated_title_for_similar(state: AgentState, *, max_len: int = 100) -> str:
    """Short title for /similar — long YAML bodies drift keyword search."""
    from bridge.legacy import ensure_legacy_backend

    ensure_legacy_backend()
    from retrieval_cascade import resolve_title

    title = resolve_title(state.query_title, state.user_query)
    base = (title or state.user_query or "").strip()
    if len(base) <= max_len:
        return base
    head = re.split(r"[?\n]", base, maxsplit=1)[0].strip()
    return (head or base)[:max_len]


def build_search_focus(
    state: AgentState,
    *,
    reason: Optional[str] = None,
) -> str:
    """Build a narrower Stack Overflow query for re-search."""
    base = (state.query_title or state.user_query).strip()
    terms = sorted(extract_key_terms(state.user_query), key=len, reverse=True)[:8]
    providers = _detect_providers(state.user_query)

    parts: list[str] = []
    if state.query_title and state.query_title.strip():
        parts.append(state.query_title.strip()[:200])
    elif base:
        parts.append(base[:200])

    if providers:
        parts.append(" ".join(providers))

    if terms:
        parts.append(" ".join(terms[:5]))

    if reason == "zero_retrieval":
        parts.append("stack overflow solution")
    elif reason == "low_domain_match":
        parts.append("accepted answer")
    elif reason == "low_faithfulness":
        parts.append("specific configuration steps")

    focus = " ".join(dict.fromkeys(" ".join(parts).split()))
    return focus[:400] if focus else state.user_query


def suggest_tier_hint(state: AgentState, *, reason: Optional[str] = None) -> str:
    """Suggest cascade entry style for the next search attempt."""
    ladder_tier = tier_for_search_attempt(state.search_attempts, state=state)
    if is_deterministic_ladder():
        return ladder_tier

    meta = state.cascade_meta or {}
    assess = state.last_assessment or {}
    mode = meta.get("retrieval_mode") or assess.get("retrieval_mode")
    plane = int(meta.get("winning_plane") or assess.get("winning_plane") or 0)
    total = len(state.raw_articles)
    low_conf = bool(
        assess.get("low_confidence")
        or meta.get("search_extension_used")
        or (mode and mode not in ("strict", None))
    )

    if total == 0 or reason == "zero_retrieval":
        return "extended_keywords"

    if reason in (
        "low_domain_match",
        "low_faithfulness",
        "low_evidence_overlap",
        "evidence_too_weak_to_ship",
        "answer_not_grounded_in_facts",
    ):
        if plane < 3 or low_conf or mode in ("title-only", "strict", None):
            return "extended_keywords"

    if reason == "no_relevant":
        return "title-only" if plane >= 2 else "extended_keywords"

    if mode and mode != "strict" and plane < 3:
        return "extended_keywords"
    return "strict"


def cascade_body_for_tier(
    state: AgentState,
    query_text: str,
    tier_hint: Optional[str],
) -> tuple[str, Optional[str]]:
    """Map tier_hint to cascade body/title inputs."""
    from bridge.legacy import ensure_legacy_backend

    ensure_legacy_backend()
    from retrieval_cascade import resolve_title

    title = resolve_title(state.query_title, state.user_query)
    hint = (tier_hint or "strict").strip().lower()

    if hint == "title-only":
        body = title or query_text[:200]
        return body, title
    if hint == "extended_keywords":
        terms = extract_key_terms(state.user_query)
        kw = " ".join(sorted(terms, key=len, reverse=True)[:12])
        body = f"{query_text} {kw}".strip()
        return body[:500], title
    if hint in ("tag_similar", "tag-scoped"):
        terms = extract_key_terms(state.user_query)
        kw = " ".join(sorted(terms, key=len, reverse=True)[:8])
        short_title = truncated_title_for_similar(state)
        return (kw or short_title or query_text[:200]), short_title or title
    return query_text, title
