"""Post-classify relevance tightening for agent mode."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "can could should may might must shall to of in for on with at by from as into "
    "through during before after above below between out off over under again "
    "how what when where why which who whom this that these those i me my we our "
    "you your they their it its".split()
)

_CLOUD_TERMS = (
    "aws", "amazon", "azure", "gcp", "google", "kubernetes", "k8s", "docker",
    "terraform", "lambda", "ec2", "s3", "eks", "ecs", "helm", "nginx", "heroku",
    "cloud", "devops", "alb", "ingress", "stack", "overflow", "flutter", "dart",
)

_DIET_TERMS = (
    "diet", "nutrition", "nutrient", "supplement", "vitamin", "mineral",
    "protein", "carbohydrate", "fiber", "omega", "dha", "epa", "glucose",
    "insulin", "diabetes", "obesity", "metabolic", "cardiovascular", "blood",
    "cholesterol", "inflammation", "gut", "microbiota", "microbiome",
    "calorie", "kcal", "dose", "dosage", "clinical", "trial", "placebo",
    "randomized", "meta-analysis", "pubmed", "human", "patient",
)


def _domain_boost_terms() -> tuple[str, ...]:
    try:
        from bridge.domain import is_dietnerd

        if is_dietnerd():
            return _DIET_TERMS
    except Exception:
        pass
    return _CLOUD_TERMS


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


def _is_code_noise(token: str) -> bool:
    """Detect user-specific code identifiers that pollute search/relevance.

    e.g. kubernetes_service.questo-server-service.metadata.0.name, URLs, resource
    paths. These are unique to the asker's config and match no general SO post.
    """
    if len(token) > 28:
        return True
    if "/" in token or ":" in token:
        return True
    # Dotted paths that include digits (metadata.0.name) or 2+ dots are code refs.
    if token.count(".") >= 2 or ("." in token and any(c.isdigit() for c in token)):
        return True
    # Long snake_case / kebab identifiers with several segments.
    if token.count("_") >= 2 or token.count("-") >= 2:
        return True
    return False


def extract_key_terms(query: str) -> set[str]:
    terms: set[str] = set()
    lower = (query or "").lower()
    for term in _domain_boost_terms():
        if term in lower:
            terms.add(term)
    for m in re.finditer(r"[a-zA-Z][a-zA-Z0-9_./:-]{2,}", query or ""):
        tok = m.group(0).lower()
        if tok in _STOPWORDS or len(tok) < 4:
            continue
        if _is_code_noise(tok):
            continue
        terms.add(tok)
    return terms


def _article_text(article: dict) -> str:
    parts = [
        str(article.get("title") or ""),
        str(article.get("abstract") or ""),
        str(article.get("summary") or ""),
        str(article.get("answer_body") or ""),
        str(article.get("body") or ""),
    ]
    return " ".join(p for p in parts if p).lower()


def term_overlap_score(query: str, article: dict) -> float:
    terms = extract_key_terms(query)
    if not terms:
        return 1.0
    text = _article_text(article)
    if not text.strip():
        return 0.0
    hits = sum(1 for t in terms if t in text)
    return hits / len(terms)


def post_filter_relevant(
    articles: List[dict],
    query: str,
    *,
    min_overlap: float | None = None,
) -> tuple[List[dict], Dict[str, Any]]:
    """Drop or rank articles by question-term overlap."""
    from agent.faith_policy import post_filter_mode
    from agent.tools.rerank import is_rerank_enabled, rerank_articles

    mode = post_filter_mode()
    threshold = min_overlap if min_overlap is not None else _env_float("AGENT_DOMAIN_OVERLAP_MIN", 0.25)
    if not articles:
        return [], {"filtered_out": 0, "threshold": threshold, "mode": mode}

    rank_fn = (
        (lambda qs, arts: rerank_articles(qs, arts))
        if is_rerank_enabled()
        else (lambda qs, arts: sorted(arts, key=lambda a: term_overlap_score(qs, a), reverse=True))
    )
    ranked = rank_fn(query, articles)
    if mode == "rank_only":
        return ranked, {
            "filtered_out": 0,
            "threshold": threshold,
            "mode": "rank_only",
            "top_overlap": round(term_overlap_score(query, ranked[0]), 3) if ranked else 0.0,
        }

    kept: List[dict] = []
    scored: List[tuple[float, dict]] = []
    for art in articles:
        score = term_overlap_score(query, art)
        scored.append((score, art))
        if score >= threshold:
            kept.append(art)

    if kept:
        kept.sort(key=lambda a: term_overlap_score(query, a), reverse=True)
        return kept, {
            "filtered_out": len(articles) - len(kept),
            "threshold": threshold,
            "mode": "filter",
            "top_overlap": round(term_overlap_score(query, kept[0]), 3),
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    fallback = [a for _, a in scored[: max(1, _env_int("AGENT_RESCUE_TOP_K", 3))]]
    return fallback, {
        "filtered_out": len(articles) - len(fallback),
        "threshold": threshold,
        "mode": "filter",
        "fallback_top_k": True,
    }


def relevance_prefilter_top_k() -> int:
    """Max articles sent to the LLM relevance classifier (v6 cost control)."""
    return max(1, _env_int("AGENT_RELEVANCE_PREFILTER_TOP_K", 20))


def prefilter_articles_for_llm(
    articles: List[dict],
    query: str,
    *,
    top_k: int | None = None,
) -> tuple[List[dict], Dict[str, Any]]:
    """Cheap term-overlap shortlist before per-article relevance LLM calls."""
    k = top_k if top_k is not None else relevance_prefilter_top_k()
    if not articles:
        return [], {"prefilter_in": 0, "prefilter_out": 0, "prefilter_top_k": k}
    if len(articles) <= k:
        return list(articles), {
            "prefilter_in": len(articles),
            "prefilter_out": len(articles),
            "prefilter_top_k": k,
            "prefilter_applied": False,
        }
    ranked = sorted(articles, key=lambda a: term_overlap_score(query, a), reverse=True)
    kept = ranked[:k]
    return kept, {
        "prefilter_in": len(articles),
        "prefilter_out": len(kept),
        "prefilter_top_k": k,
        "prefilter_applied": True,
        "top_overlap": round(term_overlap_score(query, kept[0]), 3) if kept else 0.0,
    }


def cap_relevant_articles(articles: List[dict], query: str) -> List[dict]:
    """Keep top-K by rerank or term overlap (stricter than linear cap for agent)."""
    from agent.tools.rerank import is_rerank_enabled, rerank_articles

    top_k = _env_int("AGENT_ADAPTIVE_TOP_K", 5)
    if len(articles) <= top_k:
        return articles
    if is_rerank_enabled():
        return rerank_articles(query, articles, top_k=top_k)
    ranked = sorted(articles, key=lambda a: term_overlap_score(query, a), reverse=True)
    return ranked[:top_k]


def mean_domain_overlap(query: str, articles: List[dict]) -> float:
    if not articles:
        return 0.0
    return sum(term_overlap_score(query, a) for a in articles) / len(articles)


def _dedupe_key(article: dict) -> str:
    """Identity for an article: prefer PMID / answer id, else normalized body."""
    pmid = article.get("PMID") or article.get("pmid")
    if not pmid:
        try:
            pmid = article.get("MedlineCitation", {}).get("PMID")
        except Exception:
            pmid = None
    if pmid:
        return f"pmid:{pmid}"
    aid = article.get("answer_id") or article.get("id")
    if aid:
        return f"aid:{aid}"
    body = (
        article.get("answer_body")
        or article.get("abstract")
        or article.get("body")
        or article.get("summary")
        or ""
    )
    norm = re.sub(r"\s+", " ", str(body)).strip().lower()[:200]
    if norm:
        return f"body:{norm}"
    url = article.get("answer_url") or article.get("url") or article.get("link") or ""
    return f"url:{url}"


def dedupe_articles(articles: List[dict]) -> List[dict]:
    """Drop articles with identical answer body / id so top-K slots aren't wasted."""
    seen: set[str] = set()
    out: List[dict] = []
    for art in articles or []:
        key = _dedupe_key(art)
        if key in seen:
            continue
        seen.add(key)
        out.append(art)
    return out
