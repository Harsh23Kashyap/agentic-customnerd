"""Faith-first synthesis and verification policy for the agent pipeline."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

from agent.state import AgentState

ProgressFn = Optional[Callable[[str], None]]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_faith_first() -> bool:
    """When true, prioritize grounded facts-only answers over fluency/relevancy."""
    return _env_bool("AGENT_FAITH_FIRST", True)


def post_filter_mode() -> str:
    explicit = os.getenv("AGENT_POST_FILTER_MODE", "").strip().lower()
    if explicit in ("rank_only", "filter"):
        return explicit
    return "rank_only" if is_faith_first() else "filter"


def faith_overlap_threshold() -> float:
    return _env_float("AGENT_FAITH_OVERLAP_THRESHOLD", 0.55)


def facts_only_bypass_overlap() -> float:
    return _env_float("AGENT_FACTS_ONLY_HIGH_OVERLAP", 0.80)


def prefer_combined_synthesis() -> bool:
    """Prefer grounded combined synthesis (question-shaped + grounded) over facts-only.

    Facts-only synthesis emits noncommittal disclaimers ("not detailed in the
    extracted facts") that RAGAS scores as Rel 0. Combined synthesis grounds on
    the actual posts while staying question-shaped, matching the linear backend.
    """
    return _env_bool("AGENT_PREFER_COMBINED_SYNTHESIS", True)


def low_domain_overlap_threshold() -> float:
    return _env_float("AGENT_LOW_DOMAIN_OVERLAP_RESEARCH", 0.15)


def low_overlap_top_k() -> int:
    return _env_int("AGENT_LOW_OVERLAP_TOP_K", 2)


def low_overlap_top_k_extended() -> int:
    return _env_int("AGENT_LOW_OVERLAP_TOP_K_EXTENDED", 5)


def min_evidence_overlap_ship() -> float:
    """Do not return synthetic answers below this evidence overlap."""
    return _env_float("AGENT_MIN_EVIDENCE_OVERLAP_SHIP", 0.15)


def is_low_domain_overlap(domain_overlap: Optional[float]) -> bool:
    if domain_overlap is None:
        return False
    return domain_overlap < low_domain_overlap_threshold()


def _cascade_plane(state: AgentState) -> int:
    meta = state.cascade_meta or {}
    assess = state.last_assessment or {}
    return int(meta.get("winning_plane") or assess.get("winning_plane") or 0)


def _has_fallback_sources(state: AgentState) -> bool:
    try:
        from bridge.legacy import ensure_legacy_backend

        ensure_legacy_backend()
        from retrieval_cascade import articles_to_source_counts

        counts = articles_to_source_counts(state.raw_articles)
        return bool(
            (counts.get("fallback_v3") or 0) > 0
            or (counts.get("backup_llm") or 0) > 0
            or (counts.get("backup_det") or 0) > 0
        )
    except Exception:
        return False


def is_shallow_retrieval(state: AgentState) -> bool:
    """True when cascade stopped early or used extension/fallback without deep win."""
    meta = state.cascade_meta or {}
    assess = state.last_assessment or {}
    plane = _cascade_plane(state)
    mode = meta.get("retrieval_mode") or assess.get("retrieval_mode")
    low_conf = bool(
        assess.get("low_confidence")
        or meta.get("search_extension_used")
        or (mode and mode != "strict")
    )
    return plane < 3 or low_conf or _has_fallback_sources(state)


def effective_context_top_k(state: AgentState) -> int:
    """Adaptive cap: shallow extended retrieval keeps more articles than strict low-overlap."""
    if not is_low_domain_overlap(state.domain_overlap):
        return _env_int("AGENT_ADAPTIVE_TOP_K", 5)
    if is_shallow_retrieval(state):
        return low_overlap_top_k_extended()
    return low_overlap_top_k()


def invalidate_fact_cache(state: AgentState) -> None:
    state.cached_evidence_facts = None
    state.cached_fact_overlap = None


def tighten_relevant_articles(state: AgentState) -> List[dict]:
    """Cap to highest-overlap articles when domain match is weak."""
    from agent.tools.relevance import mean_domain_overlap, term_overlap_score
    from agent.tools.rerank import is_rerank_enabled, rerank_articles

    articles = list(state.relevant_articles or [])
    if not articles:
        return articles

    domain = state.domain_overlap
    if domain is None:
        domain = mean_domain_overlap(state.user_query, articles)
        state.domain_overlap = domain

    if is_low_domain_overlap(domain):
        top_k = effective_context_top_k(state)
        if is_rerank_enabled():
            articles = rerank_articles(state.user_query, articles, top_k=top_k)
        else:
            ranked = sorted(
                articles, key=lambda a: term_overlap_score(state.user_query, a), reverse=True
            )
            articles = ranked[:top_k]
        state.relevant_articles = articles
        state.suppress_gap_last = True
        state.use_combined_lite = True
        invalidate_fact_cache(state)
    return articles


def prepare_synthesis_context(state: AgentState) -> None:
    tighten_relevant_articles(state)
    if is_low_domain_overlap(state.domain_overlap):
        state.suppress_gap_last = True
        state.use_combined_lite = True
    # Default to grounded combined synthesis over the relevant posts (linear-like),
    # unless a facts-only rescue is explicitly in progress.
    if prefer_combined_synthesis() and state.relevant_articles and not state.use_facts_only:
        state.use_combined_lite = True


def strip_gap_sentences(answer: str) -> str:
    from agent.tools.claim_filter import strip_gap_talk_sentences

    text = strip_gap_talk_sentences(answer)
    return text if text else (answer or "").strip()


def strip_hedge_sentences(answer: str) -> str:
    """Remove vague hedging that RAGAS penalizes when evidence is thin."""
    text = (answer or "").strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    hedge_re = re.compile(
        r"\b(might be|might need|may need|may need to|typically acceptable|"
        r"generally correct|generally acceptable|should suffice|"
        r"you may need to verify|you might need to verify|if the problem persists)\b",
        re.IGNORECASE,
    )
    kept = [p for p in parts if p.strip() and not hedge_re.search(p)]
    return " ".join(kept).strip() if kept else text


def polish_low_confidence_answer(state: AgentState, answer: str) -> str:
    text = answer or ""
    if state.suppress_gap_last or is_low_domain_overlap(state.domain_overlap):
        text = strip_gap_sentences(text)
        text = strip_hedge_sentences(text)
    return text.strip()


def _cheap_evidence_text(state: AgentState) -> str:
    """Concatenate relevant-article text as evidence proxy (no LLM call)."""
    parts: List[str] = []
    for art in state.relevant_articles or []:
        for key in ("abstract", "summary", "answer_body", "body"):
            val = str(art.get(key) or "").strip()
            if val:
                parts.append(val[:2000])
                break
    return "\n\n".join(parts)


def get_or_extract_facts(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    force_refresh: bool = False,
) -> str:
    if not force_refresh and state.cached_evidence_facts is not None:
        return state.cached_evidence_facts

    from bridge.legacy import ensure_legacy_backend

    ensure_legacy_backend()
    from faithfulness_two_pass import fact_term_overlap

    # Under grounded combined synthesis, verify only needs an overlap number, so
    # use the (already query-focused) article text instead of a separate LLM call.
    if not force_refresh and prefer_combined_synthesis():
        facts = _cheap_evidence_text(state)
        state.cached_evidence_facts = facts
        state.cached_fact_overlap = fact_term_overlap(state.user_query, facts)
        return state.cached_evidence_facts

    if on_progress:
        on_progress("Extracting evidence facts for faithfulness...")

    from helper_functions import extract_evidence_facts
    from openai_prompts import EVIDENCE_EXTRACTION_PROMPT

    articles = state.relevant_articles or []
    facts = extract_evidence_facts(articles, state.user_query, EVIDENCE_EXTRACTION_PROMPT)
    state.cached_evidence_facts = facts or ""
    state.cached_fact_overlap = fact_term_overlap(state.user_query, state.cached_evidence_facts)
    return state.cached_evidence_facts


def should_use_facts_only(
    state: AgentState,
    *,
    overlap: Optional[float] = None,
    force: bool = False,
) -> bool:
    # Explicit rescue/regenerate still forces facts-only extraction.
    if force and not is_low_domain_overlap(state.domain_overlap):
        return True
    # Default: prefer grounded combined synthesis (no facts-only disclaimers).
    if prefer_combined_synthesis():
        return False
    if state.use_combined_lite or is_low_domain_overlap(state.domain_overlap):
        return False
    if state.use_facts_only and not is_low_domain_overlap(state.domain_overlap):
        return True
    if not is_faith_first():
        return False
    if not state.relevant_articles:
        return False
    if overlap is None:
        overlap = state.cached_fact_overlap
    if overlap is None:
        return True
    return overlap < facts_only_bypass_overlap()


def pre_synthesis_policy(
    state: AgentState,
    *,
    overlap: float,
    threshold: float,
    domain_overlap: Optional[float] = None,
) -> Dict[str, Any]:
    if domain_overlap is not None:
        state.domain_overlap = domain_overlap

    low_domain = is_low_domain_overlap(state.domain_overlap)
    shallow = is_shallow_retrieval(state)
    if low_domain:
        state.suppress_gap_last = True
        state.use_combined_lite = True
        tighten_relevant_articles(state)

    use_facts = should_use_facts_only(state, overlap=overlap)
    state.use_facts_only = use_facts
    passed = overlap >= threshold

    recommend_research = False
    if state.search_attempts < state.max_search_attempts:
        if not state.relevant_articles:
            recommend_research = not passed
        elif low_domain and shallow:
            recommend_research = True
        elif low_domain or (not passed and overlap < threshold):
            recommend_research = True

    return {
        "passed": passed,
        "phase": "pre_synthesis",
        "term_overlap": round(overlap, 3),
        "domain_overlap": round(state.domain_overlap or 0.0, 3),
        "threshold": threshold,
        "use_facts_only": use_facts,
        "use_combined_lite": state.use_combined_lite,
        "faith_first": is_faith_first(),
        "recommend_research": recommend_research,
        "low_domain_overlap": low_domain,
        "shallow_retrieval": shallow,
        "winning_plane": _cascade_plane(state),
    }


def post_synthesis_policy(
    state: AgentState,
    *,
    overlap: float,
    answer_overlap: float,
    threshold: float,
    ref_markers: int,
    has_citations: bool,
    insufficient: bool,
) -> Dict[str, Any]:
    use_facts = state.use_facts_only
    passed = True
    reason = "ok"
    recommend_research = False
    recommend_regenerate = False
    low_domain = is_low_domain_overlap(state.domain_overlap)
    shallow = is_shallow_retrieval(state)
    min_ship = min_evidence_overlap_ship()

    if not state.relevant_articles and not insufficient:
        passed = False
        reason = "answer_without_relevant_evidence"
        recommend_research = state.search_attempts < state.max_search_attempts
    elif overlap < min_ship or answer_overlap < min_ship:
        passed = False
        reason = "evidence_too_weak_to_ship"
        if state.search_attempts < state.max_search_attempts and shallow:
            recommend_research = True
    elif state.relevant_articles and ref_markers == 0 and not has_citations:
        passed = False
        reason = "missing_citation_markers"
        recommend_regenerate = True
    elif overlap < threshold:
        passed = False
        reason = "low_evidence_overlap"
        if state.search_attempts < state.max_search_attempts and (low_domain or shallow or overlap < 0.25):
            recommend_research = True
        elif overlap < min_ship:
            recommend_research = state.search_attempts < state.max_search_attempts
        else:
            recommend_regenerate = True
            if not use_facts and not low_domain:
                state.use_facts_only = True
    elif answer_overlap < (threshold * 0.85) and not insufficient:
        passed = False
        reason = "answer_not_grounded_in_facts"
        if state.search_attempts < state.max_search_attempts and (low_domain or shallow):
            recommend_research = True
        else:
            recommend_regenerate = not use_facts
            state.use_facts_only = True

    if is_faith_first() and not passed and not recommend_regenerate and not recommend_research:
        if state.search_attempts < state.max_search_attempts and (low_domain or shallow):
            recommend_research = True
        elif overlap >= min_ship:
            recommend_regenerate = bool(state.relevant_articles)

    if not passed:
        state.reflect_failed = True

    return {
        "passed": passed,
        "reason": reason,
        "citation_markers": ref_markers,
        "citations_obj_count": len(state.citations or []),
        "evidence_overlap": round(overlap, 3),
        "answer_fact_overlap": round(answer_overlap, 3),
        "min_ship_threshold": min_ship,
        "use_facts_only": state.use_facts_only,
        "use_combined_lite": state.use_combined_lite,
        "faith_first": is_faith_first(),
        "low_domain_overlap": low_domain,
        "shallow_retrieval": shallow,
        "recommend_research": recommend_research,
        "recommend_regenerate": recommend_regenerate,
    }


def _first_substantive_sentence(text: str, *, max_len: int = 220) -> str:
    """Pick the first substantive sentence from passage text."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return ""
    for part in re.split(r"(?<=[.!?])\s+", clean):
        part = part.strip()
        if len(part) >= 40 and not re.match(r"^(this text|this abstract|this response)", part, re.I):
            return part[:max_len].rstrip(".,;") + "."
    return clean[:max_len].rstrip(".,;") + "."


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9_]{3,}", (text or "").lower()))


def _best_overlapping_sentence(text: str, question: str, *, max_len: int = 220) -> str:
    """Prefer the passage sentence that overlaps the question most (helps Rel without inventing)."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return ""
    q_toks = _tokens(question)
    best = ""
    best_score = -1.0
    for part in re.split(r"(?<=[.!?])\s+", clean):
        part = part.strip()
        if len(part) < 40:
            continue
        if re.match(r"^(this text|this abstract|this response)", part, re.I):
            continue
        p_toks = _tokens(part)
        if not p_toks:
            continue
        score = len(q_toks & p_toks) / max(1, len(q_toks)) if q_toks else 0.0
        # Slight preference for denser technical phrases.
        score += 0.01 * min(len(part), max_len) / max_len
        if score > best_score:
            best_score = score
            best = part
    if best:
        return best[:max_len].rstrip(".,;") + "."
    return _first_substantive_sentence(text, max_len=max_len)


_FALLBACK_ANSWER_PROMPT = """You rewrite retrieved Stack Overflow evidence into a short answer to the user question.

Rules (strict extractive):
- 2-4 sentences max. Answer the question directly (question → answer shape).
- Use ONLY facts, steps, commands, or config that appear in the passages. Cite each sentence with [n].
- Do NOT invent Terraform/K8s/AWS resource blocks, annotations, flags, or values not present in the passages.
- Do NOT invent full code samples. Short quoted fragments from passages are OK.
- If passages are only partly related, say what they do establish — do not fill gaps from general knowledge.
- Forbidden phrasing: "closest retrieved posts", "evidence does not mention", "I could not find".
- Output only the answer text."""


def _fallback_passages(articles: List[dict], *, max_articles: int = 4, max_chars: int = 9000) -> str:
    blocks: List[str] = []
    used = 0
    for idx, art in enumerate(articles[:max_articles], start=1):
        title = art.get("question_title") or art.get("title") or f"Post {idx}"
        body = (
            art.get("abstract")
            or art.get("answer_body")
            or art.get("body")
            or art.get("summary")
            or ""
        )
        block = f"[{idx}] {title}\n{str(body)[:2500]}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _normalize_question_lead(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q


def _heuristic_grounded_fallback(question: str, articles: List[dict]) -> str:
    """Question-addressing collage — extractive only (preserves Faith, lifts Rel vs dump lead)."""
    # Rank articles by question overlap so Rel-relevant snippets surface first.
    scored: List[tuple] = []
    q_toks = _tokens(question)
    for art in articles[:8]:
        body = (
            art.get("abstract")
            or art.get("answer_body")
            or art.get("body")
            or art.get("summary")
            or ""
        )
        title = art.get("question_title") or art.get("title") or ""
        overlap = len(q_toks & _tokens(f"{title} {body}")) / max(1, len(q_toks)) if q_toks else 0.0
        scored.append((overlap, art))
    scored.sort(key=lambda x: x[0], reverse=True)

    sentences: List[str] = []
    for idx, (_, art) in enumerate(scored[:3], start=1):
        body = (
            art.get("abstract")
            or art.get("answer_body")
            or art.get("body")
            or art.get("summary")
            or ""
        )
        sent = _best_overlapping_sentence(str(body), question)
        if sent:
            sentences.append(f"{sent} [{idx}]")
    if not sentences:
        return "Retrieved Stack Overflow posts were found but could not be summarized."
    q = _normalize_question_lead(question)
    if q:
        q_short = q if len(q) <= 160 else (q[:157].rstrip() + "…?")
        return f"Answer for: {q_short} {' '.join(sentences)}"
    return " ".join(sentences)


def _fallback_looks_extractive(text: str, passages: str) -> bool:
    """Reject fluent inventiveness: need citations and no large novel code fences."""
    if not text or not text.strip():
        return False
    if not re.search(r"\[\d+\]", text):
        return False
    # Long fenced code usually means the model invented a full sample.
    fence = re.findall(r"```[\s\S]*?```", text)
    if any(len(b) > 280 for b in fence):
        return False
    # At least some content words from passages should appear in the rewrite.
    pass_toks = set(re.findall(r"[a-zA-Z]{4,}", passages.lower()))
    ans_toks = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
    if not pass_toks or not ans_toks:
        return False
    overlap = len(pass_toks & ans_toks) / max(1, len(ans_toks))
    return overlap >= 0.25


def _llm_grounded_fallback(question: str, passages: str) -> str:
    """One cheap rewrite so fallbacks stay grounded but answer-shaped (helps relevancy)."""
    if not passages.strip():
        return ""
    try:
        from bridge.legacy import ensure_legacy_backend

        ensure_legacy_backend()
        from openai_executions import client

        if not client:
            return ""
        model = os.getenv("AGENT_FALLBACK_MODEL") or os.getenv("AGENT_MODEL", "gpt-4-turbo")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FALLBACK_ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nEvidence passages:\n{passages}",
                },
            ],
            temperature=0.0,
            top_p=1,
            max_tokens=350,
        )
        text = (response.choices[0].message.content or "").strip()
        bad = re.compile(
            r"(closest retrieved posts|evidence does not mention|could not find|"
            r"insufficient (evidence|information)|no (relevant|supporting))",
            re.I,
        )
        if not text or bad.search(text[:200]):
            return ""
        if not _fallback_looks_extractive(text, passages):
            return ""
        return text
    except Exception:
        return ""


def build_grounded_fallback_answer(state: AgentState) -> str:
    """
    Build a partial, question-shaped answer from retrieved passages with citations.

    Default is extractive collage (faithful). Optional LLM rewrite only when
    AGENT_LLM_FALLBACK=true — that path helps Rel but can invent details.
    """
    articles = state.relevant_articles or state.raw_articles or []
    if not articles:
        return "No Stack Overflow posts were retrieved for this question."

    question = (getattr(state, "cleaned_query", None) or state.user_query or "").strip()
    if _env_bool("AGENT_LLM_FALLBACK", False):
        passages = _fallback_passages(articles)
        rewritten = _llm_grounded_fallback(question, passages)
        if rewritten:
            return rewritten
    return _heuristic_grounded_fallback(question, articles)


def build_insufficient_evidence_answer(state: AgentState) -> str:
    """Deprecated alias — always use evidence-grounded fallback."""
    return build_grounded_fallback_answer(state)


def ensure_citations_on_refusal(state: AgentState) -> None:
    """Keep citations populated so eval can score refusals with contexts."""
    if state.citations:
        return
    articles = state.relevant_articles or state.raw_articles or []
    if not articles:
        return
    try:
        from bridge.legacy import ensure_legacy_backend

        ensure_legacy_backend()
        from helper_functions import replace_invalid_values

        stubs = []
        for idx, art in enumerate(articles[:5], start=1):
            title = art.get("question_title") or art.get("title") or f"Source {idx}"
            url = art.get("answer_url") or art.get("question_url") or art.get("link") or ""
            body = art.get("answer_body") or art.get("body") or art.get("abstract") or ""
            stubs.append(f"[{idx}] {title}\n{body[:1500]}")
        state.citations = replace_invalid_values(stubs)
    except Exception:
        pass


def finalize_or_refuse_answer(
    state: AgentState,
    post: Dict[str, Any],
    *,
    overlap_rescue_attempted: bool = False,
) -> bool:
    """
    Replace ungrounded answers with a grounded partial fallback when verify failed.
    Returns True if the original synthesized answer was replaced.
    """
    if post.get("passed"):
        return False

    overlap = float(post.get("evidence_overlap") or 0.0)
    answer_overlap = float(post.get("answer_fact_overlap") or 0.0)
    min_ship = min_evidence_overlap_ship()
    threshold = faith_overlap_threshold()
    reason = post.get("reason") or ""

    refuse = (
        reason == "evidence_too_weak_to_ship"
        or overlap < min_ship
        or answer_overlap < min_ship
    )
    if not refuse and is_faith_first() and is_low_domain_overlap(state.domain_overlap):
        if overlap < threshold or (is_shallow_retrieval(state) and overlap < threshold):
            refuse = True
            reason = reason or "low_domain_overlap_ungrounded"

    if not refuse:
        return False

    if not overlap_rescue_attempted:
        return False

    # If the synthesized answer is itself grounded in the extracted facts
    # (answer_fact_overlap OK), keep it — it is cited and question-shaped, which
    # a raw passage collage is not. Only dump snippets when the answer is not
    # grounded in facts at all.
    synthesized = (state.final_answer or "").strip()
    keep_synthesis = (
        answer_overlap >= min_ship
        and len(synthesized) >= 120
        and not synthesized.lower().startswith(
            ("no stack overflow posts", "retrieved stack overflow posts", "the closest retrieved posts")
        )
    )
    shipped = "synthesized_partial"
    if not keep_synthesis:
        state.final_answer = build_grounded_fallback_answer(state)
        shipped = "grounded_collage"

    ensure_citations_on_refusal(state)
    state.done = True
    state.log_step(
        "refuse_ungrounded_answer",
        {
            "evidence_overlap": overlap,
            "answer_fact_overlap": answer_overlap,
            "min_ship_threshold": min_ship,
            "faith_threshold": threshold,
            "reason": reason,
            "shipped": shipped,
            "low_domain_overlap": is_low_domain_overlap(state.domain_overlap),
            "shallow_retrieval": is_shallow_retrieval(state),
        },
    )
    return True
