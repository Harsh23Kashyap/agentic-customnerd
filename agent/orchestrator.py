"""Agent orchestrators: hybrid, parity, adaptive ReAct, and LangGraph."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from config import AGENT_MODEL, AGENT_MODE, MAX_AGENT_STEPS, MAX_SEARCH_ATTEMPTS, USE_LANGGRAPH
from agent.faith_policy import is_faith_first
from agent.state import AgentState
from agent.tools import TOOL_DEFINITIONS, dispatch_tool, tool_assess_retrieval
from agent.pipeline import run_parity_pipeline, run_enhanced_pipeline
from prompts.agent_system import AGENT_SYSTEM_PROMPT, get_agent_system_prompt

logger = logging.getLogger(__name__)

ProgressFn = Optional[Callable[[str], None]]


def _client() -> OpenAI:
    return OpenAI()


def _init_state_limits(state: AgentState) -> None:
    state.max_search_attempts = MAX_SEARCH_ATTEMPTS


def run_hybrid_agent(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> AgentState:
    """Parity-style enhanced pipeline plus optional post-verify rescue pass."""
    if on_progress:
        on_progress("AGENT_MODE=hybrid")
    return run_enhanced_pipeline(state, on_progress=on_progress, allow_rescue_pass=True)


def run_react_agent(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> AgentState:
    """ReAct loop with tool calling and safety rails."""
    client = _client()
    messages: List[dict] = [
        {"role": "system", "content": get_agent_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Question: {state.user_query}\n"
                f"Title hint: {state.query_title or '(none)'}\n"
                f"Date cutoff: {state.max_date or '(none)'}"
            ),
        },
    ]

    for step in range(1, MAX_AGENT_STEPS + 1):
        if state.done:
            break

        if on_progress:
            on_progress(f"Agent step {step}/{MAX_AGENT_STEPS}...")

        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = response.choices[0].message
        messages.append(choice.model_dump(exclude_none=True))

        tool_calls = choice.tool_calls or []
        if not tool_calls:
            _fallback_tool_chain(state, on_progress=on_progress)
            continue

        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "synthesize_answer" and not state.relevant_articles:
                assess = state.last_assessment or tool_assess_retrieval(state, on_progress=on_progress)
                if assess.get("needs_research") and state.search_attempts < state.max_search_attempts:
                    result = {"blocked": True, "reason": "no_relevant_articles_yet", "assess": assess}
                    if on_progress:
                        on_progress("Blocked synthesize: re-search recommended.")
                else:
                    if on_progress:
                        on_progress(f"Agent tool: {name}")
                    result = json.loads(dispatch_tool(name, args, state, on_progress=on_progress))
            else:
                if on_progress:
                    on_progress(f"Agent tool: {name}")
                result = json.loads(dispatch_tool(name, args, state, on_progress=on_progress))

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
            state.log_step(f"tool_{name}", result)

    if not state.done:
        _fallback_tool_chain(state, on_progress=on_progress, force_synthesize=True)

    return state


def _fallback_tool_chain(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
    force_synthesize: bool = False,
) -> None:
    from agent.tools import (
        tool_search_stackoverflow,
        tool_organize_articles,
        tool_classify_relevance,
        tool_synthesize_answer,
        tool_verify_faithfulness,
    )

    if not state.raw_articles:
        tool_search_stackoverflow(state, on_progress=on_progress)
    if state.raw_articles and not state.organized_articles:
        tool_organize_articles(state, on_progress=on_progress)
    if state.articles_for_classify and not state.relevant_articles:
        tool_classify_relevance(state, on_progress=on_progress)
        tool_assess_retrieval(state, on_progress=on_progress)
    if force_synthesize or state.relevant_articles or state.search_attempts >= state.max_search_attempts:
        if not state.done:
            tool_synthesize_answer(state, on_progress=on_progress, force_facts_only=state.use_facts_only)
            tool_verify_faithfulness(state, on_progress=on_progress, pre_synthesis=False)
    elif not state.done:
        state.final_answer = (
            "The agent could not retrieve evidence within the step budget. "
            "Please try again or rephrase your question."
        )
        state.done = True


def run_agent(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> AgentState:
    """Dispatch by AGENT_MODE: hybrid | parity | adaptive."""
    _init_state_limits(state)
    mode = AGENT_MODE

    if mode == "hybrid":
        return run_hybrid_agent(state, on_progress=on_progress)

    if mode == "parity":
        return run_parity_pipeline(state, on_progress=on_progress)

    if on_progress:
        on_progress(f"AGENT_MODE=adaptive (langgraph={USE_LANGGRAPH})")

    if USE_LANGGRAPH:
        from agent.graph import run_graph_agent

        return run_graph_agent(state, on_progress=on_progress)

    return run_react_agent(state, on_progress=on_progress)


def build_result_object(state: AgentState) -> Dict[str, Any]:
    """Match the legacy backend response shape for frontend / eval compatibility."""
    cascade_meta = state.cascade_meta or {}
    retrieval_mode = cascade_meta.get("retrieval_mode")
    source_counts: Dict[str, Any] = {}
    confidence = "low"

    if state.last_assessment:
        confidence = state.last_assessment.get("confidence", confidence)

    try:
        from bridge.legacy import ensure_legacy_backend

        ensure_legacy_backend()
        from retrieval_cascade import articles_to_source_counts, cascade_confidence

        source_counts = articles_to_source_counts(state.raw_articles)
        if not state.last_assessment:
            confidence = cascade_confidence(retrieval_mode) if retrieval_mode else (
                "high" if state.raw_articles else "low"
            )
    except Exception:
        if not state.last_assessment:
            confidence = "high" if state.raw_articles else "low"

    return {
        "end_output": state.final_answer or "",
        "final_output": state.final_answer or "",
        "citations_obj": state.citations or [],
        "retrieval_mode": retrieval_mode,
        "search_extension_used": bool(cascade_meta.get("search_extension_used")),
        "retrieval_source_counts": source_counts,
        "confidence": confidence,
        "cascade_log": cascade_meta.get("cascade_log", []),
        "extension_planes": cascade_meta.get("extension_planes", []),
        "winning_plane": cascade_meta.get("winning_plane"),
        "extension_depth": cascade_meta.get("extension_depth", 0),
        "citation_pipeline": state.citation_pipeline,
        "agent_trace": state.agent_trace,
        "agent_backend": "cloudnerd-agent-backend",
        "agent_mode": AGENT_MODE,
        "faith_first": is_faith_first(),
        "rescue_pass_used": state.rescue_pass_used,
        "last_assessment": state.last_assessment,
        "llm_calls": _llm_call_snapshot(),
        "agent_domain": _agent_domain(),
    }


def _agent_domain() -> str:
    try:
        from bridge.domain import get_domain

        return get_domain()
    except Exception:
        return os.getenv("AGENT_DOMAIN", "CloudNerd")


def _llm_call_snapshot() -> Dict[str, Any]:
    try:
        from agent import llm_meter

        return llm_meter.snapshot()
    except Exception:
        return {}
