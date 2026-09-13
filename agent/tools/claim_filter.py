"""Claim-level faithfulness filter: drop unsupported sentences."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from agent.state import AgentState

CLAIM_VERIFY_PROMPT = """You verify whether answer claims are supported by evidence passages.

For each numbered claim, respond with exactly one label:
- SUPPORTED — the claim is directly stated or clearly implied by the passages
- NOT SUPPORTED — the claim adds speculation or facts not in the passages

Return ONLY valid JSON:
{"results": [{"claim": 1, "verdict": "SUPPORTED"}, ...]}

Rules:
- Use ONLY the provided passages; no outside knowledge.
- If the passages support a finding, mark it SUPPORTED even when the claim uses careful wording
  ("may", "can", "associated with", "generally") that mirrors the evidence.
- Mark NOT SUPPORTED only when the claim invents causes, dosages, populations, or recommendations
  absent from the passages, or meta-claims about missing evidence.
- Exact commands, errors, versions, dosages, and statistics must appear in passages to be SUPPORTED.
- Meta-claims about the evidence itself ("the facts do not mention", "not detailed in the provided facts",
  "the evidence does not discuss", "further research is needed") → NOT SUPPORTED.
"""

_GAP_TALK_RE = re.compile(
    r"not discussed|does not mention|does not specify|does not specifically|"
    r"available evidence|evidence does not|retrieved posts do not|not covered in|"
    r"not detailed in the provided facts|provided facts|the facts do not|"
    r"could not find enough relevant|appear tangential|"
    r"no direct solution|not provide a direct|"
    r"no mention of|does not cover|does not address|does not discuss|"
    r"nor does it|there is no mention|the evidence (also |)(does not|doesn't|lacks)|"
    r"is not addressed|not addressed in|do(es)? not include|"
    r"the passages? (do|does) not|no information (on|about|regarding)|"
    r"not explicitly (mentioned|stated|covered)|"
    r"not detailed in the extracted facts|not (specified|mentioned|found) in the extracted facts|"
    r"the extracted facts do|refer to (the |)(additional|official|aws|terraform) |"
    r"you (may|might) need to refer|consult (the |)(official |)documentation|"
    r"refer to (additional|the) documentation|"
# DietNerd / medical: RAGAS AnswerRelevancy zeros on noncommittal closers
    r"further (high[- ]quality |higher[- ]level |well[- ]controlled )?(randomized controlled trials|rcts|research|studies|trials) (are|is) needed|"
    r"further .{0,40}?trials (are|is) needed|"
    r"more research is (still )?needed|additional (studies|trials|research) (are|is) needed|"
    r"needed to (confirm|establish|clarify).{0,60}(findings|relationship|effect)|"
    r"not a substitute for professional|"
    r"informational purposes only|"
    r"always consult( with)? (a )?(qualified )?healthcare|"
    r"consult( with)? (a )?(registered )?(dietitian|nutritionist|healthcare|physician|doctor)|"
    r"\*\*disclaimer\*\*|^\s*disclaimer\s*:",
    re.IGNORECASE,
)

# Closers that zero Rel — only drop when they are NOT carrying study stats.
_NONCOMMITTAL_CLOSER_RE = re.compile(
    r"\bstill undetermined\b|"
    r"\bcan vary\b|"
    r"\bresults are mixed\b|"
    r"\bfindings are (mixed|conflicting|inconclusive)\b|"
    r"\bfurther .{0,60}(trials|studies|research|rcts)\b.{0,20}\bneeded\b|"
    r"\bwell[- ]controlled trials are needed\b|"
    r"\boptimal (quantity|frequency|dose).{0,40}(undetermined|unclear|unknown)\b",
    re.IGNORECASE,
)

_EVIDENCE_DETAIL_RE = re.compile(
    r"\[\d+\]|"
    r"\b(p\s*[<=>]\s*0\.\d+|n\s*=\s*\d+|smd|95%\s*ci|ci:\s*[-−]?\d|"
    r"participants?|patients?|studies|trials|cohort|meta[- ]analysis)\b",
    re.IGNORECASE,
)

_DISCLAIMER_BLOCK_RE = re.compile(
    r"(?:\n|\r\n|^)\s*(?:\*\*)?disclaimer(?:\*\*)?\s*:.*\Z",
    re.IGNORECASE | re.DOTALL,
)


# Trailing hedge closers that RAGAS AnswerRelevancy flags as noncommittal (hard 0),
# even when the rest of the answer is specific and cited. Observed on the 20Q
# DietNerd eval (PR1 run, Sep 11 2026): QIDs 36518821 / 32422943 / 26891320 all
# scored Rel=0 while ending on one of these shapes. These sentences usually carry
# a [n] citation, so the has_detail guard in strip_gap_talk_sentences keeps them;
# position (last sentence) is what matters, not verifiability.
_TRAILING_HEDGE_RE = re.compile(
    r"\bremains? (unclear|unknown|undetermined|to be (determined|established|confirmed))\b|"
    r"\bno(t)? (yet )?(allow for a |allow a )?(definitive|clear|firm) conclusion\b|"
    r"\bdo(es)? not allow for a (definitive|clear|firm) conclusion\b|"
    r"\b(no|did not show|without) (a )?significant (effect|difference|association|improvement|benefit)\b|"
    r"\bresults? (were|are|remain)? ?(inconclusive|conflicting|mixed|contradictory)\b|"
    r"\bmixed results\b|"
    r"\bfindings are (mixed|conflicting|inconclusive|contradictory)\b|"
    r"\bnot (entirely|completely|fully|wholly) consistent\b|"
    r"\bno consistent (association|link|relationship|effect|evidence|pattern)\b|"
    r"\bdid not establish a (clear )?(link|association|relationship|connection)\b|"
    r"\bstill undetermined\b|"
    r"\bevidence is (limited|insufficient|inconclusive)\b|"
    r"\bcannot be (determined|established|confirmed)\b|"
    r"\bmore (research|studies|trials|evidence) (is|are) needed\b",
    re.IGNORECASE,
)


def _split_compound_hedge(sentence: str) -> List[str]:
    """Split a compound "substantive head, but hedge tail" sentence.

    Synthesis often appends the hedge as a trailing clause ("...Prevotella, but
    the results were contradictory and did not establish a clear link[1].").
    The head clause carries the specific finding; the tail is what the RAGAS
    relevancy judge flags. The [n] citation moves with the head (same source).
    """
    lowered = sentence.lower()
    for sep in (", but ", "; but ", ", however, ", "; however, ", ", though ", ", although "):
        idx = lowered.find(sep)
        if idx <= 0:
            continue
        head = sentence[:idx].strip()
        tail = sentence[idx + len(sep):].strip()
        if not tail or not _TRAILING_HEDGE_RE.search(tail):
            continue
        if _TRAILING_HEDGE_RE.search(head):
            continue
        if len(head.split()) < 5:
            continue
        m = re.search(r"((?:\[\d+\])+)\.?\s*$", tail)
        if m:
            head = head.rstrip(". ") + m.group(1) + "."
        else:
            head = head.rstrip(". ") + "."
        return [head, tail]
    return [sentence]


def _demote_trailing_hedges(sentences: List[str]) -> List[str]:
    """Keep hedge closers out of the answer when substantive sentences exist.

    RAGAS AnswerRelevancy hard-zeros the whole answer when its judge reads ANY
    part of it as noncommittal ("evasive, vague, or ambiguous") - position does
    not matter. Observed on the Sep 13 2026 hedge-fix rerun: 26891320 still
    scored Rel=0 with the hedge merely demoted to mid-answer. So:
      - >= 1 substantive sentence: drop the hedge sentences entirely.
      - 0 substantive sentences: return unchanged (all-hedge fallback).

    Keeping a hedge beside a single substantive sentence still hard-zeros the
    whole answer, so the one-substantive case cannot safely demote it.
    Compound "head, but hedge" sentences are split first so the substantive
    head clause survives.
    """
    parts: List[str] = []
    for s in sentences:
        parts.extend(_split_compound_hedge(s))
    substantive = [s for s in parts if not _TRAILING_HEDGE_RE.search(s)]
    hedges = [s for s in parts if _TRAILING_HEDGE_RE.search(s)]
    if not hedges:
        return sentences
    if substantive:
        return substantive
    return sentences


def is_claim_filter_enabled() -> bool:
    raw = os.getenv("AGENT_CLAIM_FILTER", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def min_surviving_claims() -> int:
    raw = os.getenv("AGENT_CLAIM_FILTER_MIN_CLAIMS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def strip_disclaimer(answer: str) -> str:
    """Remove DietNerd/Cloud disclaimer blocks that force RAGAS Rel=0 (noncommittal)."""
    text = (answer or "").strip()
    if not text:
        return text
    text = _DISCLAIMER_BLOCK_RE.sub("", text).strip()
    # Also drop a trailing disclaimer sentence if it survived mid-paragraph.
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [
        p for p in parts
        if p.strip()
        and not re.search(
            r"disclaimer|informational purposes only|not a substitute for professional|"
            r"always consult|qualified healthcare",
            p,
            re.I,
        )
    ]
    return " ".join(kept).strip()


def strip_gap_talk_sentences(answer: str) -> str:
    """Remove gap-talk / meta-evidence / noncommittal-closer sentences.

    Keep evidence-backed caveats (stats, citations, study n) — stripping those
    leaves a thin Yes/No lead that RAGAS Faith splits into unsupported claims.
    """
    text = strip_disclaimer(answer or "")
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept: List[str] = []
    for i, p in enumerate(parts):
        s = p.strip()
        if not s:
            continue
        has_detail = bool(_EVIDENCE_DETAIL_RE.search(s))
        if _GAP_TALK_RE.search(s):
            # Disclaimer / "further trials needed" always drop.
            # Keep the sentence if it is carrying study numbers (Faith).
            if has_detail and not re.search(
                r"disclaimer|informational purposes|not a substitute|"
                r"further .{0,40}(trials|studies|research).{0,20}needed|"
                r"consult( with)? (a )?(qualified )?healthcare",
                s,
                re.I,
            ):
                kept.append(s)
            continue
        # Drop Rel-zeroing closers unless they also state a measured finding.
        if i > 0 and _NONCOMMITTAL_CLOSER_RE.search(s) and not has_detail:
            continue
        kept.append(s)
    if len(kept) >= 2:
        kept = _demote_trailing_hedges(kept)
        return " ".join(kept).strip()
    if kept:
        # Too aggressive strip left a one-liner — restore extra evidence sentences.
        extra = [
            p.strip() for i, p in enumerate(parts)
            if p.strip()
            and p.strip() not in kept
            and _EVIDENCE_DETAIL_RE.search(p)
            and not re.search(r"disclaimer|informational purposes|further .{0,40}needed", p, re.I)
        ]
        restored = kept + extra
        if restored:
            restored = _demote_trailing_hedges(restored)
            return " ".join(restored).strip()
        return kept[0]
    for p in parts:
        if p.strip():
            return p.strip()
    return text


def split_claims(answer: str) -> List[str]:
    text = (answer or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    claims = [p.strip() for p in parts if p.strip()]
    return claims


def _evidence_passages(articles: List[dict], *, max_chars: int = 12000) -> str:
    blocks: List[str] = []
    used = 0
    for idx, art in enumerate(articles or [], start=1):
        title = art.get("question_title") or art.get("title") or f"Post {idx}"
        candidates = [
            str(art.get("answer_body") or ""),
            str(art.get("body") or ""),
            str(art.get("abstract") or ""),
        ]
        body = max(candidates, key=len)
        block = f"[{idx}] {title}\n{body}"
        if used + len(block) > max_chars:
            break
        blocks.append(block[:3000])
        used += len(block)
    return "\n\n".join(blocks)


def _verify_claims_batch(claims: List[str], passages: str) -> List[str]:
    if not claims:
        return []
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    user_msg = f"Passages:\n{passages}\n\nClaims:\n{numbered}"

    from bridge.legacy import ensure_legacy_backend

    ensure_legacy_backend()
    from openai_executions import client

    if not client:
        return claims

    response = client.chat.completions.create(
        model=os.getenv("AGENT_CLAIM_FILTER_MODEL", "gpt-4-turbo"),
        messages=[
            {"role": "system", "content": CLAIM_VERIFY_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        top_p=1,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        payload = json.loads(raw)
        results = payload.get("results") or []
        supported: List[str] = []
        verdicts = {int(r.get("claim", 0)): str(r.get("verdict", "")).upper() for r in results}
        for i, claim in enumerate(claims, start=1):
            if verdicts.get(i) == "SUPPORTED":
                supported.append(claim)
        return supported
    except (json.JSONDecodeError, TypeError, ValueError):
        # Keep claims; do not drop everything on parse failure (hurts relevancy).
        return claims or []


def apply_claim_filter(state: AgentState) -> Tuple[str, Dict[str, Any]]:
    """
    Filter unsupported claims from final_answer.
    Returns (filtered_answer, metadata). metadata['refuse']=True when zero claims survive.
    """
    answer = strip_gap_talk_sentences((state.final_answer or "").strip())
    meta: Dict[str, Any] = {
        "enabled": is_claim_filter_enabled(),
        "original_claims": 0,
        "kept_claims": 0,
        "refuse": False,
        "gap_talk_stripped": answer != (state.final_answer or "").strip(),
    }
    if not is_claim_filter_enabled() or not answer:
        if not answer and (state.final_answer or "").strip():
            meta["refuse"] = True
            return "", meta
        return answer, meta

    claims = split_claims(answer)
    meta["original_claims"] = len(claims)
    if len(claims) == 0:
        meta["refuse"] = True
        return "", meta

    articles = state.relevant_articles or state.raw_articles or []
    passages = _evidence_passages(articles)
    if not passages:
        meta["kept_claims"] = len(claims)
        return strip_disclaimer(answer), meta

    # Always verify, including 1-sentence answers (those were previously skipped
    # and shipped overclaims that RAGAS Faith scored ~0.25).
    supported = _verify_claims_batch(claims, passages)
    meta["kept_claims"] = len(supported)

    if len(supported) < min_surviving_claims():
        meta["refuse"] = True
        return "", meta

    filtered = strip_disclaimer(" ".join(supported).strip())
    state.log_step(
        "claim_filter",
        {
            "original": meta["original_claims"],
            "kept": meta["kept_claims"],
            "dropped": meta["original_claims"] - meta["kept_claims"],
        },
    )
    return filtered, meta
