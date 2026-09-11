"""Regression tests: trailing hedge closers must not end DietNerd answers.

Fixtures are the four failing answers from the 20Q DietNerd RAGAS run
(PR1 applied, Sep 11 2026, dietnerd_agent_gold_20q_pr1_evaluation.txt).
QIDs 36518821 / 32422943 / 26891320 scored Rel=0 while ending on hedge
closers that carried [n] citations (so the has_detail guard kept them).

Run: python agent/tests/test_trailing_hedge.py
"""

import importlib.util
import os
import sys
import types

# Load claim_filter.py directly, stubbing the agent package so the test runs
# without openai / backend deps (claim_filter only needs AgentState for hints).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

_agent_pkg = types.ModuleType("agent")
_agent_pkg.__path__ = [os.path.join(_ROOT, "agent")]
sys.modules["agent"] = _agent_pkg
_state_mod = types.ModuleType("agent.state")


class AgentState:  # minimal stand-in for type hints
    pass


_state_mod.AgentState = AgentState
sys.modules["agent.state"] = _state_mod

_spec = importlib.util.spec_from_file_location(
    "agent.tools.claim_filter", os.path.join(_ROOT, "agent", "tools", "claim_filter.py")
)
_cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cf)

_TRAILING_HEDGE_RE = _cf._TRAILING_HEDGE_RE
strip_disclaimer = _cf.strip_disclaimer
strip_gap_talk_sentences = _cf.strip_gap_talk_sentences

ANS_36518821 = (
    "Yes, folic acid supplementation has shown a positive effect on cognitive tests, "
    "which may include memory performance[1][2]. Folic acid supplementation proved to "
    "have better outcomes on cognitive tests than their respective control groups in a "
    "systematic review[2]. Additionally, the combined supplementation of folic acid and "
    "vitamin B12 showed some discrepancies between studies, indicating variability in "
    "outcomes but still suggesting potential benefits[2]. However, it is important to "
    "note that the results of B-vitamin supplementation, including folic acid, in stroke "
    "survivors did not show a significant effect on cognitive outcomes[1]."
)

ANS_32422943 = (
    "Yes, there is an association between vitamin D deficiency and erectile dysfunction "
    "(ED), particularly in severe forms of ED[2]. A meta-analysis found that patients "
    "with vitamin D deficiency had significantly worse scores on the International Index "
    "of Erectile Function (IIEF) compared to controls, and this association remained "
    "significant even when considering only eugonadal patients[2]. Additionally, "
    "eugonadal patients with severe ED had lower levels of 25-hydroxyvitamin D compared "
    "to those with mild ED[2]. However, it is important to note that the overall quality "
    "and heterogeneity of clinical trials evaluating the effects of vitamin D on "
    "erectile function do not allow for a definitive conclusion[1]."
)

ANS_26891320 = (
    "Yes, higher consumption of animal flesh foods is associated with better iron "
    "status among adults in developed countries[1]. This conclusion is supported by a "
    "systematic review that analyzed both experimental and observational studies, "
    "finding that five out of seven high-quality studies showed a positive association "
    "between animal flesh intake (ranging from 85 to 300 grams per day) and iron "
    "status[1]. However, the optimal quantity or frequency of flesh intake required to "
    "maintain or achieve a healthy iron status remains unclear[1]."
)

# Faith=0 QID: no hedge closer; strip must leave the substantive answer intact.
ANS_24815945 = (
    "Yes, red and processed meat intake is associated with obesity. These findings "
    "suggest a significant relationship between the consumption of red and processed "
    "meats and increased risk of obesity[1]."
)

DISCLAIMER_ANSWER = (
    "Vitamin D may support bone health in older adults[1].\n\n"
    "**Disclaimer**: This is for informational purposes only. Always consult a "
    "qualified healthcare professional."
)


def _last_sentence(text: str) -> str:
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    return parts[-1] if parts else ""


def run() -> int:
    failures = []

    def check(name: str, cond: bool, detail: str = ""):
        if not cond:
            failures.append(f"{name}: {detail}")

    # 1-3: hedge closer must not be the final sentence after the strip,
    #      and its content must survive (Faithfulness preservation).
    for qid, ans in [
        ("36518821", ANS_36518821),
        ("32422943", ANS_32422943),
        ("26891320", ANS_26891320),
    ]:
        out = strip_gap_talk_sentences(ans)
        check(
            f"{qid} trailing hedge demoted",
            not _TRAILING_HEDGE_RE.search(_last_sentence(out)),
            f"last sentence still a hedge: {_last_sentence(out)!r}",
        )
        check(
            f"{qid} hedge content preserved",
            "[1]" in out and len(out) >= len(ans) * 0.8,
            f"content lost: {out!r}",
        )

    # 4: clean short answer passes through unchanged.
    out = strip_gap_talk_sentences(ANS_24815945)
    check("24815945 passthrough", out == ANS_24815945, f"got: {out!r}")

    # 5: pure disclaimer block still removed entirely.
    out = strip_disclaimer(DISCLAIMER_ANSWER)
    check(
        "disclaimer stripped",
        "disclaimer" not in out.lower() and "informational purposes" not in out.lower(),
        f"got: {out!r}",
    )
    check("disclaimer body kept", out.startswith("Vitamin D may support"), f"got: {out!r}")

    # 6: an all-hedge answer must not crash and must return something.
    out = strip_gap_talk_sentences("The evidence is limited. Results are inconclusive.")
    check("all-hedge fallback", bool(out.strip()), "empty output")

    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print(" -", f)
        return 1
    print("OK - 6 trailing-hedge regression tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
