"""Agent tools wrapping the legacy CloudNerd retrieval pipeline."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from bridge.legacy import ensure_legacy_backend
from bridge.collect import collect_with_fallback
from agent.state import AgentState
from agent.tools.organize import tool_organize_articles, summarize_relevant_articles
from agent.tools.assess import (
    tool_assess_retrieval,
    tool_verify_faithfulness,
    tool_reflect_on_answer,
)
from agent.tools.relevance import (
    post_filter_relevant,
    cap_relevant_articles,
    prefilter_articles_for_llm,
    dedupe_articles,
)
from agent.tools.search_focus import build_search_focus, cascade_body_for_tier, tier_for_search_attempt, is_deterministic_ladder
from agent.faith_policy import (
    get_or_extract_facts,
    invalidate_fact_cache,
    polish_low_confidence_answer,
    prepare_synthesis_context,
    should_use_facts_only,
    strip_gap_sentences,
)

ProgressFn = Optional[Callable[[str], None]]


def _emit(on_progress: ProgressFn, message: str) -> None:
    if on_progress:
        on_progress(message)


def tool_search_stackoverflow(
    state: AgentState,
    *,
    search_focus: Optional[str] = None,
    tier_hint: Optional[str] = None,
    on_progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Run retrieval against the active domain corpus (SO cascade or PubMed legacy)."""
    ensure_legacy_backend()
    from bridge.domain import is_dietnerd

    diet = is_dietnerd()
    corpus_label = "PubMed" if diet else "Stack Overflow"

    state.search_attempts += 1
    query_text = (search_focus or state.user_query).strip()
    if not search_focus and state.last_assessment:
        query_text = state.last_assessment.get("suggested_search_focus") or query_text
    if not tier_hint:
        tier_hint = tier_for_search_attempt(state.search_attempts, state=state) if is_deterministic_ladder() else None
    if not tier_hint and state.last_assessment:
        tier_hint = state.last_assessment.get("suggested_tier_hint")

    body_text, title = cascade_body_for_tier(state, query_text, tier_hint)
    tier_only = tier_hint if is_deterministic_ladder() else None

    _emit(
        on_progress,
        f"Agent: searching {corpus_label} ({body_text[:80]}...) "
        f"tier={tier_hint or ('legacy' if diet else 'strict')}",
    )

    def collect_wrapper(query_list, max_date=None, **kwargs):
        return collect_with_fallback(query_list, max_date=max_date, **kwargs)

    planes: List[dict] = []

    def on_plane(plane: dict) -> None:
        planes.append(plane)
        flag = "WIN" if plane.get("success") else (
            "EXTEND" if plane.get("extension_triggered") else "try"
        )
        _emit(
            on_progress,
            f"Cascade plane {plane.get('plane')}: {plane.get('tier')}/{plane.get('step')} "
            f"[{flag}] queries={plane.get('queries_count')} articles={plane.get('articles_found')}",
        )

    # DietNerd / PubMed: always legacy query_generation + Entrez collect.
    # SO cascade assumes Stack Overflow tags, clean_query JSON, and SO schemas.
    use_cascade = False
    cascade_meta: Dict[str, Any] = {}
    articles: List[dict] = []
    if not diet:
        from retrieval_cascade import (
            cascade_collect_articles,
            is_cascade_mode,
            articles_to_source_counts,
        )

        use_cascade = is_cascade_mode()
        if use_cascade:
            articles, cascade_meta = cascade_collect_articles(
                body_text,
                title,
                state.max_date,
                collect_wrapper,
                on_plane=on_plane,
                tier_only=tier_only,
            )
            try:
                from retrieval_gap_fill import gap_fill_retrieval

                articles, gap_meta = gap_fill_retrieval(
                    articles,
                    body_text,
                    title,
                    state.max_date,
                    collect_wrapper,
                    cascade_meta,
                )
                if gap_meta.get("gap_fill_used"):
                    cascade_meta["gap_fill"] = gap_meta
            except ImportError:
                pass

    if diet or not use_cascade:
        from helper_functions import query_generation

        _, _, query_list = query_generation(body_text)
        articles = collect_wrapper(query_list, max_date=state.max_date) or []
        cascade_meta = {
            "retrieval_mode": "legacy",
            "extension_planes": planes,
            "domain": "DietNerd" if diet else "CloudNerd",
            "corpus": corpus_label,
        }

    def _article_key(art: dict) -> str:
        if not isinstance(art, dict):
            return str(id(art))
        # PubMed Entrez XML (pre-organize)
        try:
            pmid = art.get("MedlineCitation", {}).get("PMID")
            if pmid:
                return f"pmid:{pmid}"
        except Exception:
            pass
        for k in ("PMID", "pmid", "id", "answer_id", "question_id", "link", "url", "title"):
            val = art.get(k) if isinstance(art, dict) else None
            if val:
                return f"{k}:{val}"
        return str(id(art))

    seen: set[str] = set()
    merged: List[dict] = []
    prior_keys = {_article_key(a) for a in (state.raw_articles or [])}
    new_only = 0
    for art in state.raw_articles + (articles or []):
        key = _article_key(art)
        if key in seen:
            continue
        seen.add(key)
        if key not in prior_keys:
            new_only += 1
        merged.append(art)

    found_n = len(articles or [])
    # Empty / duplicate-only search: keep prior organized+relevant (avoid re-LLM).
    skip_reprocess = (
        found_n == 0 or new_only == 0
    ) and bool(state.organized_articles)

    state.raw_articles = merged
    state.cascade_meta = cascade_meta or {}
    if not skip_reprocess:
        state.organized_articles = []
        state.relevant_articles = []
        invalidate_fact_cache(state)

    source_counts: Dict[str, Any] = {}
    try:
        from retrieval_cascade import articles_to_source_counts

        source_counts = articles_to_source_counts(merged) if merged else {}
    except Exception:
        source_counts = {"pubmed": len(merged)} if diet else {}

    state.log_step(
        "search_stackoverflow",
        {
            "found": found_n,
            "new_unique": new_only,
            "total": len(merged),
            "attempt": state.search_attempts,
            "tier_hint": tier_hint or ("legacy" if diet else "strict"),
            "search_focus": query_text[:120],
            "corpus": corpus_label,
            "domain": "DietNerd" if diet else "CloudNerd",
            "skipped_reprocess": skip_reprocess,
        },
    )

    return {
        "articles_this_search": found_n,
        "new_unique": new_only,
        "total_articles": len(merged),
        "retrieval_mode": cascade_meta.get("retrieval_mode") if cascade_meta else "legacy",
        "source_counts": source_counts,
        "search_attempts": state.search_attempts,
        "tier_hint": tier_hint or ("legacy" if diet else "strict"),
        "corpus": corpus_label,
        "skipped_reprocess": skip_reprocess,
    }


def tool_classify_relevance(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Run relevance classification with post-filter and cap."""
    ensure_legacy_backend()
    from helper_functions import concurrent_relevance_classification

    articles = state.articles_for_classify
    if not articles:
        state.relevant_articles = []
        return {"relevant_count": 0, "message": "No articles to classify."}

    shortlist, prefilter_meta = prefilter_articles_for_llm(articles, state.user_query)
    _emit(
        on_progress,
        f"Agent: classifying {len(shortlist)}/{len(articles)} articles for relevance "
        f"(prefilter_top_k={prefilter_meta.get('prefilter_top_k')})...",
    )
    relevant, irrelevant = concurrent_relevance_classification(shortlist, state.user_query)
    del irrelevant

    state.relevant_before_rescue = len(relevant)
    state.rescued_count = 0

    try:
        from citation_rescue import is_citation_rescue_enabled, rescue_relevant_articles

        # Prefer shortlist pool first; fall back to full organized set if empty.
        rescue_pool = shortlist or articles
        if is_citation_rescue_enabled() and not relevant and rescue_pool:
            rescued = rescue_relevant_articles(rescue_pool, state.user_query)
            if rescued:
                relevant = rescued
                state.rescued_count = len(rescued)
                state.cascade_meta["citation_rescue"] = {"rescued_count": state.rescued_count}
    except ImportError:
        pass

    deduped = dedupe_articles(relevant or [])
    dupes_removed = len(relevant or []) - len(deduped)
    filtered, filter_meta = post_filter_relevant(deduped, state.user_query)
    capped = cap_relevant_articles(filtered, state.user_query)
    if dupes_removed:
        filter_meta["dupes_removed"] = dupes_removed

    if not capped and state.raw_articles:
        from agent.tools.rerank import is_rerank_enabled, rerank_articles
        from retrieval_rank import rank_articles as overlap_rank

        floor_k = 2
        pool = dedupe_articles(state.raw_articles)
        if is_rerank_enabled():
            capped = rerank_articles(state.user_query, pool, top_k=floor_k)
        else:
            capped = overlap_rank(state.user_query, pool, top_k=floor_k)
        filter_meta["retrieval_floor"] = True

    state.relevant_articles = capped
    state.log_step(
        "classify_relevance",
        {
            "relevant_count": len(capped),
            "before_filter": len(relevant or []),
            **prefilter_meta,
            **filter_meta,
        },
    )

    return {
        "relevant_count": len(capped),
        "raw_count": len(articles),
        "llm_classified": len(shortlist),
        "rescued_count": state.rescued_count,
        "filtered_out": filter_meta.get("filtered_out", 0),
        **prefilter_meta,
    }


def tool_synthesize_answer(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    force_facts_only: bool = False,
) -> Dict[str, Any]:
    """Generate final cited answer from relevant articles."""
    ensure_legacy_backend()
    from helper_functions import (
        generate_final_response,
        trim_relevant_articles_by_token_limit,
        print_referenced_articles,
        replace_invalid_values,
    )

    if not state.relevant_articles and state.rescued_count == 0:
        if state.search_attempts < state.max_search_attempts:
            return {
                "skipped": True,
                "reason": "no_relevant_articles",
                "message": "Re-search recommended before synthesizing.",
            }

    _emit(on_progress, "Agent: synthesizing final response...")

    prepare_synthesis_context(state)

    articles = trim_relevant_articles_by_token_limit(state.relevant_articles, state.user_query)
    articles = cap_relevant_articles(articles, state.user_query)

    try:
        from citation_rescue import cap_articles_for_final_response, _final_context_top_k

        top_k = _final_context_top_k()
        if top_k > 0 and len(articles) > top_k:
            articles = cap_articles_for_final_response(articles, state.user_query, top_k=top_k)
    except ImportError:
        pass

    if not articles:
        from agent.faith_policy import build_grounded_fallback_answer, ensure_citations_on_refusal

        state.final_answer = build_grounded_fallback_answer(state)
        ensure_citations_on_refusal(state)
        state.done = True
        state.log_step("synthesize_answer", {"empty": True, "fallback": "grounded_partial"})
        return {"answer_length": len(state.final_answer), "citations": len(state.citations or [])}

    # Query-focused summaries on the final small set so RAGAS contexts align with
    # the question (linear-quality). Guard flag keeps regenerate/rescue calls free.
    if not any(a.get("_qsummarized") for a in articles):
        articles = summarize_relevant_articles(articles, state.user_query, on_progress=on_progress)
        for a in articles:
            a["_qsummarized"] = True
        state.relevant_articles = articles

    use_facts = should_use_facts_only(state, force=force_facts_only)
    use_combined = state.use_combined_lite and not use_facts

    try:
        from faithfulness_two_pass import is_two_pass_enabled, run_two_pass_final_response

        if use_facts:
            facts = get_or_extract_facts(state, on_progress=on_progress, force_refresh=True)
            final_result = generate_final_response(facts, state.user_query, use_extracted_facts=True)
        elif use_combined:
            final_result = generate_final_response(
                articles, state.user_query, use_combined_lite=True,
            )
        elif is_two_pass_enabled():
            final_result = run_two_pass_final_response(articles, state.user_query)
        else:
            final_result = generate_final_response(articles, state.user_query)
    except ImportError:
        if use_facts:
            facts = get_or_extract_facts(state, on_progress=on_progress, force_refresh=True)
            final_result = generate_final_response(facts, state.user_query, use_extracted_facts=True)
        elif use_combined:
            final_result = generate_final_response(
                articles, state.user_query, use_combined_lite=True,
            )
        else:
            final_result = generate_final_response(articles, state.user_query)

    if state.suppress_gap_last:
        final_result = polish_low_confidence_answer(state, final_result or "")

    # Disclaimer / "further research needed" closers make RAGAS AnswerRelevancy = 0
    # (noncommittal). Strip before claim filter + shipping.
    from agent.tools.claim_filter import strip_gap_talk_sentences

    final_result = strip_gap_talk_sentences(final_result or "")

    citation_generation = print_referenced_articles(final_result, articles)

    try:
        from citation_rescue import ensure_citations, pipeline_stats

        citation_generation = ensure_citations(final_result, articles, citation_generation)
        state.citation_pipeline = pipeline_stats(
            organized=len(state.organized_articles or state.raw_articles),
            relevant_before_rescue=state.relevant_before_rescue,
            relevant_after_rescue=len(state.relevant_articles),
            cited=len(citation_generation or []),
            rescued=state.rescued_count,
        )
    except ImportError:
        pass

    state.final_answer = final_result
    state.citations = replace_invalid_values(citation_generation)
    state.done = True
    state.log_step(
        "synthesize_answer",
        {"citations": len(state.citations or []), "facts_only": use_facts, "combined_lite": use_combined},
    )

    return {
        "answer_length": len(final_result or ""),
        "citations": len(state.citations or []),
        "facts_only": use_facts,
    }


TOOL_DEFINITIONS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_stackoverflow",
            "description": "Search Stack Overflow using the CloudNerd cascade retriever.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_focus": {
                        "type": "string",
                        "description": "Optional refined query. Defaults to the user question.",
                    },
                    "tier_hint": {
                        "type": "string",
                        "description": "Optional hint: strict, title-only, or extended_keywords.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_articles",
            "description": "Structure and enrich retrieved Stack Overflow posts before relevance filtering.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_relevance",
            "description": "Filter retrieved articles for relevance to the user question.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_retrieval",
            "description": "Check retrieval quality (counts, confidence, domain overlap) before deciding to re-search or answer.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_answer",
            "description": "Produce the final cited answer from relevant articles.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_faithfulness",
            "description": "Verify the answer is grounded in retrieved evidence; may trigger re-search or facts-only regeneration.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def dispatch_tool(
    name: str,
    arguments: Dict[str, Any],
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> str:
    """Execute a tool and return JSON string result for the model."""
    if name == "search_stackoverflow":
        result = tool_search_stackoverflow(
            state,
            search_focus=arguments.get("search_focus"),
            tier_hint=arguments.get("tier_hint"),
            on_progress=on_progress,
        )
    elif name == "organize_articles":
        result = tool_organize_articles(state, on_progress=on_progress)
    elif name == "classify_relevance":
        result = tool_classify_relevance(state, on_progress=on_progress)
    elif name == "assess_retrieval":
        result = tool_assess_retrieval(state, on_progress=on_progress)
    elif name == "synthesize_answer":
        result = tool_synthesize_answer(
            state,
            on_progress=on_progress,
            force_facts_only=state.use_facts_only,
        )
    elif name in ("verify_faithfulness", "reflect_on_answer"):
        result = tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)
