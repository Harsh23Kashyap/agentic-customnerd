"""Agent session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    session_id: str
    user_query: str
    query_title: Optional[str] = None
    max_date: Optional[str] = None

    raw_articles: List[dict] = field(default_factory=list)
    organized_articles: List[dict] = field(default_factory=list)
    relevant_articles: List[dict] = field(default_factory=list)
    cascade_meta: Dict[str, Any] = field(default_factory=dict)
    agent_trace: List[dict] = field(default_factory=list)

    final_answer: Optional[str] = None
    citations: List[Any] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None

    search_attempts: int = 0
    max_search_attempts: int = 2
    reflect_failed: bool = False
    rescue_pass_used: bool = False
    use_facts_only: bool = False

    relevant_before_rescue: int = 0
    rescued_count: int = 0
    citation_pipeline: Optional[Dict[str, Any]] = None
    last_assessment: Optional[Dict[str, Any]] = None
    cached_evidence_facts: Optional[str] = None
    cached_fact_overlap: Optional[float] = None
    faith_regenerate_attempts: int = 0
    domain_overlap: Optional[float] = None
    suppress_gap_last: bool = False
    use_combined_lite: bool = False

    def log_step(self, step: str, detail: Any = None) -> None:
        entry = {"step": step}
        if detail is not None:
            entry["detail"] = detail
        self.agent_trace.append(entry)

    @property
    def articles_for_classify(self) -> List[dict]:
        return self.organized_articles if self.organized_articles else self.raw_articles

    def clear_for_rescue_pass(self) -> None:
        """Reset synthesis output before a hybrid rescue pass."""
        self.done = False
        self.final_answer = None
        self.citations = []
        self.raw_articles = []
        self.relevant_articles = []
        self.organized_articles = []
        self.reflect_failed = False
        self.cached_evidence_facts = None
        self.cached_fact_overlap = None
        self.faith_regenerate_attempts = 0
        self.domain_overlap = None
        self.suppress_gap_last = False
        self.use_combined_lite = False
        self.search_attempts = 0
