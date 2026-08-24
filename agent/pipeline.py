"""Enhanced agent pipeline with assess, verify, and optional rescue."""

from __future__ import annotations

from typing import Callable, Optional

from config import HYBRID_RESCUE_ENABLED
from agent.faith_policy import (
    is_faith_first,
    is_low_domain_overlap,
    finalize_or_refuse_answer,
    min_evidence_overlap_ship,
    build_grounded_fallback_answer,
    ensure_citations_on_refusal,
)
from agent.state import AgentState
from agent.tools import (
    tool_search_stackoverflow,
    tool_organize_articles,
    tool_classify_relevance,
    tool_assess_retrieval,
    tool_synthesize_answer,
)
from agent.tools.assess import tool_verify_faithfulness
from agent.tools.search_focus import build_search_focus, tier_for_search_attempt
from agent.tools.claim_filter import apply_claim_filter, is_claim_filter_enabled, min_surviving_claims

ProgressFn = Optional[Callable[[str], None]]


def _hybrid_rescue_on_zero_only() -> bool:
    """Read at call time so variables.env overrides apply after load_environment()."""
    import os

    raw = os.getenv("AGENT_HYBRID_RESCUE_ON_ZERO_ONLY", "false")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _search_classify_assess(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    search_focus: str | None = None,
    tier_hint: str | None = None,
) -> dict:
    if tier_hint is None:
        tier_hint = tier_for_search_attempt(state.search_attempts, state=state)
    search_result = tool_search_stackoverflow(
        state,
        search_focus=search_focus,
        tier_hint=tier_hint,
        on_progress=on_progress,
    )
    if not search_result.get("skipped_reprocess"):
        tool_organize_articles(state, on_progress=on_progress)
        tool_classify_relevance(state, on_progress=on_progress)
    return tool_assess_retrieval(state, on_progress=on_progress)


def _research_with_focus(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    reason: str | None = None,
    assess: dict | None = None,
) -> dict:
    tier = tier_for_search_attempt(state.search_attempts, state=state)
    if on_progress:
        on_progress(f"Re-searching (tier={tier}, reason={reason or 'low_domain_match'})...")
    return _search_classify_assess(
        state,
        on_progress=on_progress,
        search_focus=(assess or {}).get("suggested_search_focus")
        or build_search_focus(state, reason=reason or "low_domain_match"),
        tier_hint=tier,
    )


def _is_shippable_synthesis(text: str) -> bool:
    """A synthesized answer worth keeping over a raw snippet dump."""
    t = (text or "").strip()
    if len(t) < 80:
        return False
    low = t.lower()
    refusal_leads = (
        "no stack overflow posts",
        "retrieved stack overflow posts were found",
        "the closest retrieved posts",
        "i could not find",
        "insufficient evidence",
        "answer for:",
    )
    if any(low.startswith(r) for r in refusal_leads):
        return False
    # Need at least one substantive sentence so it reads as an answer.
    import re as _re

    sentences = [s for s in _re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) >= 30]
    return len(sentences) >= 1


def _ship_grounded_fallback(
    state: AgentState,
    *,
    step: str,
    meta: dict | None = None,
    prefer_synthesis: bool = True,
) -> None:
    """
    Ship a partial answer when verify/claim-filter failed.

    Prefer the synthesized, cited answer (question-shaped → good relevancy) with
    gap-talk stripped; only fall back to a raw passage collage when no usable
    synthesis exists.
    """
    from agent.tools.claim_filter import strip_gap_talk_sentences

    synthesized = strip_gap_talk_sentences(state.final_answer or "")
    if prefer_synthesis and _is_shippable_synthesis(synthesized):
        state.final_answer = synthesized
        ensure_citations_on_refusal(state)
        state.done = True
        state.log_step(
            step,
            {**(meta or {}), "shipped": "synthesized_partial"},
        )
        return

    state.final_answer = build_grounded_fallback_answer(state)
    ensure_citations_on_refusal(state)
    state.done = True
    state.log_step(step, {**(meta or {}), "shipped": "grounded_collage"})


def _apply_claim_filter_step(state: AgentState, *, on_progress: ProgressFn = None) -> bool:
    """Run claim filter; return True if pipeline should stop (fallback shipped)."""
    if not is_claim_filter_enabled() or not state.final_answer:
        return False
    if on_progress:
        on_progress("Claim filter: verifying sentences against evidence...")
    # Preserve the synthesized answer before the filter may blank it out.
    synthesized_before = (state.final_answer or "").strip()
    filtered, meta = apply_claim_filter(state)
    if meta.get("refuse"):
        # Keep the synthesized answer if the filter emptied it (evidence too thin
        # to verify sentence-by-sentence, but the answer is still cited + on-topic).
        if synthesized_before and not (state.final_answer or "").strip():
            state.final_answer = synthesized_before
        if on_progress:
            on_progress("Claim filter: shipping partial answer...")
        _ship_grounded_fallback(state, step="claim_filter_grounded_fallback", meta=meta)
        return True
    if meta.get("kept_claims", 0) and filtered and filtered != state.final_answer:
        state.final_answer = filtered
        state.log_step("claim_filter_applied", meta)
    elif filtered:
        state.final_answer = filtered
    return False


def _attempt_overlap_rescue(state: AgentState, *, on_progress: ProgressFn = None) -> bool:
    """Re-synthesize facts-only and claim-filter before grounded fallback."""
    if on_progress:
        on_progress("Overlap rescue: facts-only re-synthesis from top passages...")
    state.done = False
    state.final_answer = None
    state.use_facts_only = True
    synth = tool_synthesize_answer(state, on_progress=on_progress, force_facts_only=True)
    if synth.get("skipped") or not (state.final_answer or "").strip():
        return False

    filtered, meta = apply_claim_filter(state)
    if meta.get("refuse") or meta.get("kept_claims", 0) < min_surviving_claims():
        state.log_step("overlap_rescue_insufficient", meta)
        return False

    if filtered:
        state.final_answer = filtered
    state.done = True
    state.log_step("overlap_rescue_success", meta)
    return True


def _maybe_refuse_with_rescue(
    state: AgentState,
    post: dict,
    *,
    on_progress: ProgressFn = None,
) -> bool:
    """Try overlap rescue, then grounded fallback if verify still failed."""
    if post.get("passed"):
        return False

    if _attempt_overlap_rescue(state, on_progress=on_progress):
        post = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
        if post.get("passed"):
            return True

    if finalize_or_refuse_answer(state, post, overlap_rescue_attempted=True):
        if on_progress:
            on_progress("Shipping grounded partial answer (verify failed after rescue).")
        return True
    return False


def run_enhanced_pipeline(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    allow_rescue_pass: bool = False,
) -> AgentState:
    """search -> classify -> assess -> synthesize -> verify, with optional re-search/rescue."""
    if on_progress:
        on_progress("Enhanced pipeline: search -> assess -> synthesize -> verify")

    assess = _search_classify_assess(state, on_progress=on_progress)

    if assess.get("needs_research") and state.search_attempts < state.max_search_attempts:
        if on_progress:
            on_progress("Re-searching with refined focus...")
        assess = _research_with_focus(
            state,
            on_progress=on_progress,
            reason=assess.get("rescue_reason"),
            assess=assess,
        )

    pre = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=True)
    if (
        pre.get("recommend_research")
        or (
            is_low_domain_overlap(state.domain_overlap)
            and state.search_attempts < state.max_search_attempts
        )
    ) and state.search_attempts < state.max_search_attempts:
        assess = _research_with_focus(
            state,
            on_progress=on_progress,
            reason="low_domain_match",
            assess=assess,
        )
        pre = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=True)

    synth = tool_synthesize_answer(
        state,
        on_progress=on_progress,
        force_facts_only=bool(state.use_facts_only or pre.get("use_facts_only")),
    )

    if synth.get("skipped") and state.search_attempts < state.max_search_attempts:
        assess = _research_with_focus(
            state,
            on_progress=on_progress,
            reason=assess.get("rescue_reason"),
            assess=assess,
        )
        tool_synthesize_answer(state, on_progress=on_progress, force_facts_only=False)

    if _apply_claim_filter_step(state, on_progress=on_progress):
        return state

    post = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)

    # Under grounded combined synthesis a regenerate just repeats the same call;
    # skip it once we already hold a shippable grounded answer (saves LLM calls).
    from agent.faith_policy import prefer_combined_synthesis

    grounded_ready = (
        prefer_combined_synthesis()
        and _is_shippable_synthesis((state.final_answer or "").strip())
        and float(post.get("answer_fact_overlap") or 0.0) >= min_evidence_overlap_ship()
    )
    max_faith_regen = 0 if grounded_ready else 1
    while state.faith_regenerate_attempts < max_faith_regen:
        if post.get("recommend_research") and state.search_attempts < state.max_search_attempts:
            if on_progress:
                on_progress("Post-verify: re-searching due to weak evidence overlap...")
            assess = _research_with_focus(
                state,
                on_progress=on_progress,
                reason=post.get("reason") or "low_domain_match",
                assess=assess,
            )
            state.faith_regenerate_attempts += 1
            state.done = False
            state.final_answer = None
            tool_synthesize_answer(state, on_progress=on_progress, force_facts_only=False)
            if _apply_claim_filter_step(state, on_progress=on_progress):
                return state
            post = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
            break

        if not post.get("recommend_regenerate"):
            break

        state.faith_regenerate_attempts += 1
        if on_progress:
            on_progress(
                f"Faithfulness regenerate {state.faith_regenerate_attempts}/{max_faith_regen}..."
            )
        state.done = False
        state.final_answer = None
        tool_synthesize_answer(
            state,
            on_progress=on_progress,
            force_facts_only=bool(state.use_facts_only and not state.use_combined_lite),
        )
        if _apply_claim_filter_step(state, on_progress=on_progress):
            return state
        post = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)

    min_ship = min_evidence_overlap_ship()
    post_overlap = float(post.get("evidence_overlap") or 0.0)
    too_weak_for_rescue = (
        not post.get("passed")
        and post_overlap < min_ship
        and (post.get("reason") or "") == "evidence_too_weak_to_ship"
    )

    if too_weak_for_rescue and _maybe_refuse_with_rescue(state, post, on_progress=on_progress):
        return state

    rescue_trigger = (
        post.get("recommend_research")
        or assess.get("needs_rescue")
        or not post.get("passed")
        or (is_faith_first() and state.reflect_failed)
    )
    if _hybrid_rescue_on_zero_only():
        # v6: avoid repeating full search+organize on soft faith fails
        rescue_trigger = bool(
            not state.relevant_articles
            or assess.get("rescue_reason") in ("zero_retrieval", "no_relevant")
            or (post.get("reason") or "") == "evidence_too_weak_to_ship"
            or too_weak_for_rescue
        )

    if (
        allow_rescue_pass
        and HYBRID_RESCUE_ENABLED
        and not state.rescue_pass_used
        and rescue_trigger
    ):
        if on_progress:
            on_progress("Hybrid rescue pass (deep extended search)...")
        reason = post.get("reason") or assess.get("rescue_reason") or "low_domain_match"
        state.rescue_pass_used = True
        state.clear_for_rescue_pass()
        rescue_assess = _search_classify_assess(
            state,
            on_progress=on_progress,
            search_focus=build_search_focus(state, reason=reason),
            tier_hint="extended_keywords",
        )
        tool_synthesize_answer(state, on_progress=on_progress, force_facts_only=False)
        if _apply_claim_filter_step(state, on_progress=on_progress):
            return state
        post = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
        if post.get("recommend_regenerate") and post.get("passed"):
            state.done = False
            state.final_answer = None
            tool_synthesize_answer(state, on_progress=on_progress, force_facts_only=False)
            if _apply_claim_filter_step(state, on_progress=on_progress):
                return state
            post = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
        state.last_assessment = rescue_assess

    if not post.get("passed"):
        _maybe_refuse_with_rescue(state, post, on_progress=on_progress)

    return state


def run_parity_pipeline(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> AgentState:
    """Deterministic enhanced pipeline without hybrid rescue pass."""
    if on_progress:
        on_progress("Parity mode: enhanced pipeline")
    return run_enhanced_pipeline(state, on_progress=on_progress, allow_rescue_pass=False)
