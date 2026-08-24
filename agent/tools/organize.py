"""Organize and URL-enrich retrieved articles (linear pipeline parity).

v6: AGENT_FAST_ORGANIZE maps Stack Overflow payloads to the article schema
without per-article LLM calls (the dominant cost in v5).
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

from bridge.legacy import ensure_legacy_backend
from agent.state import AgentState

ProgressFn = Optional[Callable[[str], None]]


def _emit(on_progress: ProgressFn, message: str) -> None:
    if on_progress:
        on_progress(message)


def _env_bool(key: str, default: bool = True) -> bool:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_fast_organize_enabled() -> bool:
    return _env_bool("AGENT_FAST_ORGANIZE", True)


def _looks_like_pubmed(article: Any) -> bool:
    if not isinstance(article, dict):
        return False
    if "MedlineCitation" in article:
        return True
    src = str(article.get("retrieval_source") or "").lower()
    return src in ("pubmed", "ncbi", "entrez")


def _entrez_text(value: Any) -> str:
    """Normalize Bio.Entrez StringElement / list AbstractText to plain str."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("#text") or item.get("text") or item))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        return str(value.get("#text") or value.get("text") or value).strip()
    return str(value).strip()


def _fast_organize_pubmed_article(article: dict) -> Dict[str, Any]:
    """Map Entrez PubmedArticle XML dict → article schema with 0 LLM calls."""
    mc = article.get("MedlineCitation") or {}
    art = mc.get("Article") or {}
    journal = art.get("Journal") or {}
    pmid = _entrez_text(mc.get("PMID") or article.get("PMID") or article.get("pmid"))
    title = _entrez_text(art.get("ArticleTitle") or article.get("title"))
    abstract = _entrez_text(
        (art.get("Abstract") or {}).get("AbstractText")
        if isinstance(art.get("Abstract"), dict)
        else art.get("Abstract")
    )
    if not abstract:
        abstract = _entrez_text(article.get("abstract") or article.get("summary"))
    journal_title = _entrez_text(
        journal.get("Title") if isinstance(journal, dict) else journal
    )
    # Authors
    authors = []
    author_list = art.get("AuthorList") or []
    if isinstance(author_list, list):
        for a in author_list[:8]:
            if not isinstance(a, dict):
                continue
            last = _entrez_text(a.get("LastName"))
            initials = _entrez_text(a.get("Initials") or a.get("ForeName"))
            if last:
                authors.append(f"{last} {initials}".strip())
    # Date
    date = ""
    try:
        d = ((journal.get("JournalIssue") or {}).get("PubDate") or {})
        y = _entrez_text(d.get("Year"))
        m = _entrez_text(d.get("Month"))
        day = _entrez_text(d.get("Day"))
        date = "-".join(p for p in (y, m, day) if p)
    except Exception:
        date = ""
    doi = ""
    try:
        for id_obj in (article.get("PubmedData") or {}).get("ArticleIdList") or []:
            id_str = _entrez_text(id_obj)
            attrs = getattr(id_obj, "attributes", None) or {}
            if str(attrs.get("IdType", "")).lower() == "doi" or id_str.startswith("10."):
                doi = id_str
                break
    except Exception:
        pass
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    body = abstract or title
    summary = _truncate(body, 500) if body else title
    return {
        "title": title,
        "publication_type": "study",
        "url": url,
        "abstract": (abstract or title)[:4000],
        "citations": "",
        "author_name": ", ".join(authors),
        "summary": summary,
        "is_relevant": True,
        "id": pmid,
        "PMID": pmid,
        "pmid": pmid,
        "doi": doi,
        "date": date,
        "journal": journal_title or "PubMed",
        "retrieval_source": "pubmed",
        "retrieval_tier": str(article.get("retrieval_tier") or "legacy"),
        "answer_body": body,
    }


def is_organize_relevant_enabled() -> bool:
    """Query-focused summary of the final relevant posts (1 LLM call each).

    CloudNerd: compress noisy SO bodies so RAGAS contexts stay on-question.
    DietNerd: skip — PubMed abstracts *are* the evidence. Overwriting them with
    a 2–4 sentence summary makes RAGAS Faith fail when the answer copies
    sample sizes / stats that the summary dropped.
    """
    try:
        from bridge.domain import is_dietnerd

        if is_dietnerd():
            return False
    except Exception:
        pass
    return _env_bool("AGENT_ORGANIZE_RELEVANT", True)


_RELEVANT_SUMMARY_PROMPT_CLOUD = (
    "You summarize a Stack Overflow post as evidence for a specific question.\n"
    "Write 2-4 sentences focused on the concrete solution, steps, commands, or "
    "config in the post that bear on the question. Use only the post content; do "
    "not add outside facts and do not say what is missing. Output only the summary."
)

_RELEVANT_SUMMARY_PROMPT_DIET = (
    "You summarize a PubMed / nutrition research article as evidence for a specific "
    "question.\n"
    "Write 2-4 sentences focused on the study population, intervention/exposure, "
    "key findings, dosages if present, and safety notes that bear on the question. "
    "Use only the article content; do not add outside facts and do not say what is "
    "missing. Prefer human evidence. Output only the summary."
)


def _relevant_summary_prompt() -> str:
    try:
        from bridge.domain import is_dietnerd

        if is_dietnerd():
            return _RELEVANT_SUMMARY_PROMPT_DIET
    except Exception:
        pass
    return _RELEVANT_SUMMARY_PROMPT_CLOUD


def _summarize_one(article: dict, query: str) -> dict:
    body = str(
        article.get("answer_body")
        or article.get("abstract")
        or article.get("body")
        or article.get("summary")
        or ""
    ).strip()
    if len(body) < 40:
        return article
    try:
        from openai_executions import client

        if not client:
            return article
        model = os.getenv("AGENT_ORGANIZE_MODEL") or os.getenv("AGENT_MODEL", "gpt-4-turbo")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _relevant_summary_prompt()},
                {"role": "user", "content": f"Question:\n{query}\n\nEvidence:\n{body[:4000]}"},
            ],
            temperature=0.0,
            top_p=1,
            max_tokens=220,
        )
        summary = (resp.choices[0].message.content or "").strip()
        if summary:
            enriched = dict(article)
            original_abs = str(
                article.get("answer_body") or article.get("abstract") or body or ""
            ).strip()
            enriched["summary"] = summary
            # Keep the full source text as citation/RAGAS context. Replacing
            # abstract with a short summary is a Faith leak (answer cites
            # numbers the summary dropped).
            if original_abs and len(original_abs) >= len(summary):
                enriched["abstract"] = original_abs[:4000]
            else:
                enriched["abstract"] = summary
            return enriched
    except Exception:
        pass
    return article


def summarize_relevant_articles(
    articles: List[dict],
    query: str,
    *,
    on_progress: ProgressFn = None,
) -> List[dict]:
    """Attach a query-focused abstract to each (already small) relevant article."""
    if not articles or not is_organize_relevant_enabled():
        return articles
    _emit(on_progress, f"Summarizing {len(articles)} relevant articles for query focus...")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(6, len(articles))) as executor:
        return list(executor.map(lambda a: _summarize_one(a, query), articles))


def is_skip_url_reenrich_enabled() -> bool:
    return _env_bool("AGENT_SKIP_URL_REENRICH", True)


def organize_max_articles() -> int:
    return max(1, _env_int("AGENT_ORGANIZE_MAX_ARTICLES", 20))


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _TAG_RE.sub(" ", text or "").strip()


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip() + "..."


def _looks_like_so(article: Any) -> bool:
    if not isinstance(article, dict):
        return False
    src = str(article.get("retrieval_source") or "").lower()
    if src in ("stackoverflow", "stack_overflow", "so"):
        return True
    if article.get("answer_body") and (article.get("answer_url") or article.get("answer_id")):
        return True
    if article.get("question_title") and article.get("answer_body"):
        return True
    return False


def _fast_organize_so_article(article: dict) -> Dict[str, Any]:
    """Map a raw SO answer dict to DEFAULT_ARTICLE-shaped output without LLM."""
    body = _strip_html(str(article.get("answer_body") or article.get("body") or ""))
    title = str(
        article.get("question_title")
        or article.get("title")
        or ""
    ).strip()
    url = str(article.get("answer_url") or article.get("url") or article.get("link") or "")
    aid = article.get("answer_id") or article.get("id") or ""
    summary = _truncate(body, 500) if body else title

    return {
        "title": title,
        "publication_type": "study",
        "url": url,
        "abstract": body[:4000],
        "citations": "",
        "author_name": str(article.get("author_name") or ""),
        "summary": summary,
        "is_relevant": True,
        "id": str(aid),
        "doi": "",
        "date": str(article.get("date") or article.get("creation_date") or ""),
        "journal": "Stack Overflow",
        "retrieval_source": str(article.get("retrieval_source") or "stackoverflow"),
        "retrieval_tier": str(article.get("retrieval_tier") or ""),
        "answer_body": body,
        "score": article.get("score"),
        "is_accepted": article.get("is_accepted"),
    }


def cap_raw_articles_for_organize(articles: List[dict], query: str) -> List[dict]:
    """Keep top-N raw articles by cheap term overlap before organize."""
    from agent.tools.relevance import term_overlap_score

    max_n = organize_max_articles()
    if len(articles) <= max_n:
        return articles
    ranked = sorted(articles, key=lambda a: term_overlap_score(query, a), reverse=True)
    return ranked[:max_n]


def tool_organize_articles(
    state: AgentState,
    *,
    on_progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Structure raw SO posts; fast path avoids per-article LLM calls (v6)."""
    ensure_legacy_backend()
    from helper_functions import (
        concurrent_organize_database_articles,
        process_articles_by_url,
    )

    if not state.raw_articles:
        state.organized_articles = []
        return {"organized_count": 0, "message": "No articles to organize."}

    capped = cap_raw_articles_for_organize(list(state.raw_articles), state.user_query)
    _emit(
        on_progress,
        f"Organizing {len(capped)}/{len(state.raw_articles)} articles "
        f"(fast={is_fast_organize_enabled()})...",
    )

    if is_fast_organize_enabled():
        fast: List[dict] = []
        slow: List[dict] = []
        for art in capped:
            if _looks_like_so(art):
                fast.append(_fast_organize_so_article(art))
            elif _looks_like_pubmed(art):
                fast.append(_fast_organize_pubmed_article(art))
            else:
                slow.append(art)

        organized = list(fast)
        if slow:
            _emit(on_progress, f"LLM-organize fallback for {len(slow)} non-mapped article(s)...")
            organized.extend(concurrent_organize_database_articles(slow, state.user_query) or [])

        if not is_skip_url_reenrich_enabled():
            organized = process_articles_by_url(organized or [])

        state.organized_articles = organized or []
        state.log_step(
            "organize_articles",
            {
                "organized_count": len(state.organized_articles),
                "fast_organize": True,
                "fast_count": len(fast),
                "llm_organize": len(slow),
                "capped_from": len(state.raw_articles),
                "skip_url_reenrich": is_skip_url_reenrich_enabled(),
            },
        )
        return {
            "organized_count": len(state.organized_articles),
            "fast_organize": True,
            "llm_organize": len(slow),
        }

    organized = concurrent_organize_database_articles(capped, state.user_query)
    if not is_skip_url_reenrich_enabled():
        organized = process_articles_by_url(organized or [])
    state.organized_articles = organized or []
    state.log_step(
        "organize_articles",
        {
            "organized_count": len(state.organized_articles),
            "fast_organize": False,
            "llm_organize": len(state.organized_articles),
            "capped_from": len(state.raw_articles),
        },
    )
    return {"organized_count": len(state.organized_articles), "fast_organize": False}
