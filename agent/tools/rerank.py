"""Embedding-based reranking for retrieved Stack Overflow articles."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from agent.tools.relevance import term_overlap_score


def is_rerank_enabled() -> bool:
    raw = os.getenv("AGENT_RERANK", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def rerank_model_name() -> str:
    return os.getenv("AGENT_RERANK_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(rerank_model_name())


def _article_text(art: dict[str, Any]) -> str:
    parts = [
        str(art.get("question_title") or art.get("title") or ""),
        str(art.get("abstract") or art.get("summary") or ""),
        str(art.get("answer_body") or art.get("body") or ""),
    ]
    text = " ".join(p for p in parts if p).strip()
    return text[:4000] if text else ""


def rerank_score(query: str, art: dict[str, Any]) -> float:
    """Cosine similarity between query and article; term overlap as tie-break."""
    if not is_rerank_enabled():
        return term_overlap_score(query, art)
    body = _article_text(art)
    if not body:
        return term_overlap_score(query, art)
    try:
        model = _load_model()
        from sentence_transformers import util

        q_emb = model.encode(query or "", convert_to_tensor=True, show_progress_bar=False)
        a_emb = model.encode(body, convert_to_tensor=True, show_progress_bar=False)
        cos = float(util.cos_sim(q_emb, a_emb)[0][0])
        overlap = term_overlap_score(query, art)
        return cos + (overlap * 0.05)
    except Exception as exc:
        print(f"[rerank] embedding failed ({exc}); using term overlap")
        return term_overlap_score(query, art)


def rerank_articles(
    query: str,
    articles: List[dict],
    *,
    top_k: Optional[int] = None,
) -> List[dict]:
    if not articles:
        return []
    ranked = sorted(articles, key=lambda a: rerank_score(query, a), reverse=True)
    if top_k is not None and top_k > 0:
        return ranked[:top_k]
    return ranked


def rerank_meta(query: str, articles: List[dict]) -> Dict[str, Any]:
    if not articles:
        return {"enabled": is_rerank_enabled(), "top_score": 0.0}
    top = rerank_score(query, articles[0]) if articles else 0.0
    ranked = rerank_articles(query, articles, top_k=1)
    if ranked:
        top = rerank_score(query, ranked[0])
    return {"enabled": is_rerank_enabled(), "top_score": round(top, 4)}
