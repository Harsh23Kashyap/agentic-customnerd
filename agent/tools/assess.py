"""Retrieval assessment and faithfulness verification tools."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Optional

from bridge.legacy import ensure_legacy_backend
from agent.state import AgentState
from agent.tools.relevance import extract_key_terms, mean_domain_overlap
from agent.tools.search_focus import build_search_focus, suggest_tier_hint
from agent.faith_policy import (
    faith_overlap_threshold,
    get_or_extract_facts,
    post_synthesis_policy,
    pre_synthesis_policy,
)

ProgressFn = Optional[Callable[[str], None]]


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _emit(on_progress: ProgressFn, message: str) -> None:
    if on_progress:
        on_progress(message)


def _articles_with_answer_body(articles: list) -> int:
    count = 0
    for art in articles:
        body = art.get("answer_body") or art.get("body") or art.get("abstract") or ""
        if str(body).strip():
            count += 1
    return count


def tool_assess_retrieval(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Summarize retrieval quality and recommend next action."""
    ensure_legacy_backend()
    from retrieval_cascade import articles_to_source_counts, cascade_confidence

    cascade_meta = state.cascade_meta or {}
    retrieval_mode = cascade_meta.get("retrieval_mode")
    total_raw = len(state.raw_articles)
    total_org = len(state.organized_articles or state.raw_articles)
    relevant = len(state.relevant_articles)
    winning_plane = cascade_meta.get("winning_plane") or 0

    try:
        source_counts = articles_to_source_counts(state.raw_articles)
        confidence = cascade_confidence(retrieval_mode) if retrieval_mode else (
            "high" if total_raw else "low"
        )
    except Exception:
        source_counts = {}
        confidence = "high" if total_raw else "low"

    relevance_density = (relevant / total_org) if total_org else 0.0
    domain_overlap = mean_domain_overlap(state.user_query, state.relevant_articles or state.organized_articles or state.raw_articles)
    min_relevant = _env_int("AGENT_MIN_RELEVANT", 2)
    answer_body_count = _articles_with_answer_body(state.organized_articles or state.raw_articles)

    low_confidence = (
        confidence == "low"
        or (winning_plane and winning_plane >= 3)
        # "legacy" is the DietNerd/PubMed path — not a weak cascade plane.
        or (retrieval_mode and retrieval_mode not in ("strict", "legacy", None))
    )

    rescue_reason: Optional[str] = None
    needs_research = False

    if total_raw == 0:
        needs_research = True
        rescue_reason = "zero_retrieval"
    elif relevant == 0:
        needs_research = state.search_attempts < state.max_search_attempts
        rescue_reason = "no_relevant"
    elif relevant < min_relevant and low_confidence and state.search_attempts < state.max_search_attempts:
        needs_research = True
        rescue_reason = "low_relevant_count"
    elif relevant >= min_relevant and domain_overlap < float(os.getenv("AGENT_DOMAIN_OVERLAP_MIN", "0.25")):
        rescue_reason = "low_domain_match"
        needs_rescue = True

    low_domain_thresh = float(os.getenv("AGENT_LOW_DOMAIN_OVERLAP_RESEARCH", "0.15"))
    if (
        relevant > 0
        and domain_overlap < low_domain_thresh
        and state.search_attempts < state.max_search_attempts
    ):
        needs_research = True
        rescue_reason = rescue_reason or "low_domain_match"

    plane = winning_plane or 0
    # SO cascade only: low-confidence early planes may warrant another search.
    # DietNerd/legacy already did a full PubMed pull — don't force a 2nd pass
    # just because confidence heuristics were written for SO tiers.
    if (
        retrieval_mode not in ("legacy",)
        and relevant > 0
        and plane < 3
        and low_confidence
        and state.search_attempts < state.max_search_attempts
    ):
        needs_research = True
        rescue_reason = rescue_reason or "low_domain_match"

    needs_rescue = bool(
        rescue_reason
        and not needs_research
        and (low_confidence or domain_overlap < 0.35 or relevant < min_relevant)
    )

    result = {
        "total_articles": total_org,
        "relevant_count": relevant,
        "retrieval_mode": retrieval_mode,
        "winning_plane": winning_plane,
        "search_extension_used": bool(cascade_meta.get("search_extension_used")),
        "confidence": confidence,
        "low_confidence": low_confidence,
        "source_counts": source_counts,
        "search_attempts": state.search_attempts,
        "relevance_density": round(relevance_density, 3),
        "domain_overlap": round(domain_overlap, 3),
        "answer_body_count": answer_body_count,
        "needs_research": needs_research,
        "needs_rescue": needs_rescue,
        "rescue_reason": rescue_reason,
        "suggested_search_focus": build_search_focus(state, reason=rescue_reason) if rescue_reason else None,
        "suggested_tier_hint": suggest_tier_hint(state, reason=rescue_reason) if rescue_reason else "strict",
    }

    state.last_assessment = result
    state.domain_overlap = domain_overlap
    _emit(
        on_progress,
        f"Assess: relevant={relevant}/{total_org} density={relevance_density:.2f} "
        f"domain={domain_overlap:.2f} confidence={confidence}",
    )
    state.log_step("assess_retrieval", result)
    return result


def tool_verify_faithfulness(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    pre_synthesis: bool = False,
) -> Dict[str, Any]:
    """Verify answer grounding via fact extraction and term overlap."""
    enabled = os.getenv("FAITHFULNESS_VERIFY_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not enabled and not pre_synthesis:
        return {"passed": True, "skipped": True, "reason": "verify_disabled"}

    ensure_legacy_backend()
    from faithfulness_two_pass import fact_term_overlap

    if pre_synthesis:
        if not state.relevant_articles:
            return {"passed": False, "reason": "no_relevant_articles", "recommend_research": True}
        try:
            facts = get_or_extract_facts(state, on_progress=on_progress)
            overlap = fact_term_overlap(state.user_query, facts)
            threshold = faith_overlap_threshold()
            result = pre_synthesis_policy(
                state,
                overlap=overlap,
                threshold=threshold,
                domain_overlap=state.domain_overlap,
            )
            state.log_step("verify_faithfulness", result)
            return result
        except Exception as exc:
            return {"passed": True, "skipped": True, "reason": f"pre_verify_error:{exc}"}

    answer = (state.final_answer or "").strip()
    if not answer:
        return {"passed": False, "reason": "empty_answer", "recommend_research": True}

    _emit(on_progress, "Verifying answer faithfulness...")

    lower = answer.lower()
    insufficient = any(
        p in lower for p in ("could not find", "insufficient", "no relevant", "not enough evidence")
    )
    ref_markers = len(re.findall(r"\[\d+\]", answer))
    has_citations = bool(state.citations)

    overlap = 1.0
    answer_overlap = 1.0
    try:
        facts = get_or_extract_facts(state, on_progress=on_progress)
        from faithfulness_two_pass import fact_term_overlap

        overlap = fact_term_overlap(state.user_query, facts)
        answer_overlap = fact_term_overlap(answer, facts)
        threshold = faith_overlap_threshold()
    except Exception:
        threshold = faith_overlap_threshold()

    result = post_synthesis_policy(
        state,
        overlap=overlap,
        answer_overlap=answer_overlap,
        threshold=threshold,
        ref_markers=ref_markers,
        has_citations=has_citations,
        insufficient=insufficient,
    )
    state.log_step("verify_faithfulness", result)
    return result


def tool_reflect_on_answer(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for verify_faithfulness."""
    return tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
