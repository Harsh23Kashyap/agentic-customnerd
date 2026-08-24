"""LangGraph state machine for adaptive agent mode."""

from __future__ import annotations

import logging
from typing import Callable, Optional, TypedDict

from agent.state import AgentState
from agent.tools import (
    tool_search_stackoverflow,
    tool_organize_articles,
    tool_classify_relevance,
    tool_assess_retrieval,
    tool_synthesize_answer,
)
from agent.tools.assess import tool_verify_faithfulness

logger = logging.getLogger(__name__)

ProgressFn = Optional[Callable[[str], None]]


class GraphState(TypedDict, total=False):
    agent_state: AgentState
    phase: str


def _wrap(state: AgentState) -> GraphState:
    return {"agent_state": state, "phase": "start"}


def run_graph_agent(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> AgentState:
    """Run LangGraph workflow; falls back to ReAct if langgraph unavailable."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        logger.warning("langgraph not installed; falling back to ReAct adaptive loop")
        from agent.orchestrator import run_react_agent

        return run_react_agent(state, on_progress=on_progress)

    def retrieve_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        assess = s.last_assessment or {}
        if s.search_attempts >= s.max_search_attempts and s.raw_articles:
            return {**gs, "phase": "organize"}
        tool_search_stackoverflow(
            s,
            on_progress=on_progress,
            search_focus=assess.get("suggested_search_focus"),
            tier_hint=assess.get("suggested_tier_hint"),
        )
        return {**gs, "agent_state": s, "phase": "organize"}

    def organize_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        tool_organize_articles(s, on_progress=on_progress)
        return {**gs, "agent_state": s, "phase": "classify"}

    def classify_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        tool_classify_relevance(s, on_progress=on_progress)
        return {**gs, "agent_state": s, "phase": "assess"}

    def assess_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        result = tool_assess_retrieval(s, on_progress=on_progress)
        if result.get("needs_research") and s.search_attempts < s.max_search_attempts:
            return {**gs, "agent_state": s, "phase": "retrieve"}
        return {**gs, "agent_state": s, "phase": "verify_pre"}

    def verify_pre_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        pre = tool_verify_faithfulness(s, on_progress=on_progress, pre_synthesis=True)
        if pre.get("recommend_research") and s.search_attempts < s.max_search_attempts:
            return {**gs, "agent_state": s, "phase": "retrieve"}
        return {**gs, "agent_state": s, "phase": "synthesize"}

    def synthesize_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        tool_synthesize_answer(s, on_progress=on_progress, force_facts_only=s.use_facts_only)
        return {**gs, "agent_state": s, "phase": "verify_post"}

    def verify_post_node(gs: GraphState) -> GraphState:
        s = gs["agent_state"]
        result = tool_verify_faithfulness(s, on_progress=on_progress, pre_synthesis=False)
        if result.get("recommend_regenerate") and not s.use_facts_only:
            s.use_facts_only = True
            s.done = False
            s.final_answer = None
            return {**gs, "agent_state": s, "phase": "synthesize"}
        if result.get("recommend_research") and s.search_attempts < s.max_search_attempts:
            s.done = False
            s.final_answer = None
            s.relevant_articles = []
            s.organized_articles = []
            return {**gs, "agent_state": s, "phase": "retrieve"}
        return {**gs, "agent_state": s, "phase": "done"}

    def route_phase(gs: GraphState, mapping: dict) -> str:
        return mapping.get(gs.get("phase", ""), "end")

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("organize", organize_node)
    graph.add_node("classify", classify_node)
    graph.add_node("assess", assess_node)
    graph.add_node("verify_pre", verify_pre_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("verify_post", verify_post_node)

    try:
        from langgraph.graph import START

        graph.add_edge(START, "retrieve")
    except ImportError:
        graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "organize")
    graph.add_edge("organize", "classify")
    graph.add_edge("classify", "assess")
    graph.add_conditional_edges(
        "assess",
        lambda gs: "retrieve" if gs.get("phase") == "retrieve" else "verify_pre",
        {"retrieve": "retrieve", "verify_pre": "verify_pre"},
    )
    graph.add_conditional_edges(
        "verify_pre",
        lambda gs: "retrieve" if gs.get("phase") == "retrieve" else "synthesize",
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", "verify_post")
    graph.add_conditional_edges(
        "verify_post",
        lambda gs: gs.get("phase", "done"),
        {
            "retrieve": "retrieve",
            "synthesize": "synthesize",
            "done": END,
        },
    )

    compiled = graph.compile()
    final = compiled.invoke(_wrap(state))
    return final["agent_state"]
