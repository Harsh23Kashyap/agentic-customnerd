"""Synthesis instructions shared by the agent fallback and the legacy final-response prompt.

The linear backend owns ``FINAL_RESPONSE_PROMPT`` (and combined-lite / facts-only
variants) under customnerd-backend. This module does not edit that tree. At
synthesize time the same rule is appended in memory so a conflicting evidence
set cannot collapse into a flat Yes or No.
"""

from __future__ import annotations

SYNTHESIS_CONFLICT_RULE = """When the cited studies disagree, say so in the first sentence and cite both sides.
Lead with the disagreement, for example: "Possibly, but the evidence is mixed: ...".
Never give a flat Yes or No when the evidence conflicts.
Keep mixed, conflicting, inconsistent, heterogeneous, inconclusive, and "not all studies" statements in the answer, and cite the supporting side and the opposing side."""

_CONFLICT_MARKER = "When the cited studies disagree"

# Legacy modules patched at synthesize time. helper_functions binds
# FINAL_RESPONSE_PROMPT via import *, so the same attribute is updated on both.
_SYNTHESIS_PROMPT_MODULES = (
    "openai_prompts",
    "helper_functions",
    "generic_prompts",
)

# Explicit registry. A name listed here receives the conflict rule even when it
# does not match the heuristic in _is_synthesis_prompt_name (no PROMPT suffix,
# and no FINAL_RESPONSE / COMBINED / FACTS_ONLY / SYNTHESIS token). Add those
# prompts with register_synthesis_prompt so they are not skipped silently.
_SYNTHESIS_PROMPT_NAMES = {
    "FINAL_RESPONSE_PROMPT",
    "COMBINED_LITE_PROMPT",
    "COMBINED_RESPONSE_PROMPT",
    "COMBINED_LITE_RESPONSE_PROMPT",
    "FACTS_ONLY_PROMPT",
    "FACTS_ONLY_RESPONSE_PROMPT",
    "EXTRACTED_FACTS_RESPONSE_PROMPT",
    "TWO_PASS_FINAL_PROMPT",
}

# Answer prompts defined in this repo. install_synthesis_conflict_rule does not
# see them; each constant must embed SYNTHESIS_CONFLICT_RULE itself.
_LOCAL_SYNTHESIS_PROMPTS = (
    ("agent.faith_policy", "_FALLBACK_ANSWER_PROMPT"),
)


def register_synthesis_prompt(name: str) -> str:
    """Register a synthesis prompt attribute that must receive the conflict rule.

    ``install_synthesis_conflict_rule`` patches legacy prompt modules by
    attribute name. Names that match the built-in heuristic are picked up on
    their own. A prompt whose name does not match has to be registered here,
    or the rule is never appended.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("synthesis prompt name is required")
    upper = cleaned.upper()
    _SYNTHESIS_PROMPT_NAMES.add(upper)
    return upper


def _is_synthesis_prompt_name(name: str) -> bool:
    upper = name.upper()
    if upper in _SYNTHESIS_PROMPT_NAMES:
        return True
    if not upper.endswith("PROMPT"):
        return False
    return any(token in upper for token in ("FINAL_RESPONSE", "COMBINED", "FACTS_ONLY", "SYNTHESIS"))


def install_synthesis_conflict_rule() -> str:
    """Append the mixed-evidence lead rule onto already-imported synthesis prompts.

    ``helper_functions`` binds ``FINAL_RESPONSE_PROMPT`` with ``import *``, so the
    prompts module and the helper module both need the suffix. A second call
    does not duplicate the rule. Selection is the explicit registry plus the
    name heuristic; call ``register_synthesis_prompt`` for a prompt the
    heuristic does not recognize.
    """
    import sys

    rule = SYNTHESIS_CONFLICT_RULE.strip()
    for mod_name in _SYNTHESIS_PROMPT_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if not isinstance(val, str) or not _is_synthesis_prompt_name(attr):
                continue
            if _CONFLICT_MARKER in val:
                continue
            setattr(mod, attr, val.rstrip() + "\n\n" + rule)
    return rule
