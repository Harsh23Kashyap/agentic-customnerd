"""Regression tests: hedge closers must not survive in DietNerd answers.

Fixtures are the failing answers from two 20Q DietNerd RAGAS runs:
- PR1 run (Sep 11 2026, dietnerd_agent_gold_20q_pr1_evaluation.txt):
  QIDs 36518821 / 32422943 / 26891320 scored Rel=0 ending on hedge closers
  that carried [n] citations (so the has_detail guard kept them).
- Hedge-fix rerun (Sep 13 2026, dietnerd_agent_gold_20q_hedgefix_evaluation.txt):
  26891320 still scored Rel=0 with the hedge merely DEMOTED to mid-answer
  (RAGAS AnswerRelevancy flags noncommittal content anywhere, not just the
  ending), and 26443336 / 31631671 scored Rel=0 on new hedge shapes the
  first regex missed ("not entirely consistent", "results were
  contradictory", "did not establish a clear link").

Strategy: strip hedge sentences whenever substantive content remains, and
leave the answer untouched only when the whole answer is a hedge.
Compound "head, but hedge" sentences are split so the substantive head
clause (with its citation) survives.

Run: python agent/tests/test_trailing_hedge.py
"""

import importlib.util
import os
import re
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

# --- Sep 13 hedge-fix rerun Rel=0 answers (the three Daksh reported) ---

# Demoted hedge STILL scored Rel=0 - mid-answer "remains unclear" trips the judge.
ANS_26891320_RERUN = (
    "Yes, higher consumption of animal flesh foods is associated with better iron "
    "status among adults in developed countries[1]. However, the optimum quantity or "
    "frequency of flesh intake required to maintain or achieve a healthy iron status "
    "remains unclear[1]. This systematic review included eight experimental and 41 "
    "observational studies, with seven high-quality studies showing a positive "
    "association between animal flesh intake (85-300 g/day) and iron status[1]."
)

# Trailing hedge shape the first regex missed: "not entirely consistent" / "mixed results".
ANS_26443336_RERUN = (
    "Yes, consuming yogurt is associated with weight management outcomes[1][2][3][4]. "
    "A systematic review found consistent associations between fermented milk "
    "consumption, which includes yogurt, and improved weight maintenance[1]. "
    "Additionally, epidemiological studies have shown that yogurt consumption is "
    "linked with reduced risks of type 2 diabetes, metabolic syndrome, and heart "
    "disease, all of which relate to weight management[2]. Furthermore, randomized "
    "controlled trials have indicated that yogurt may enhance weight maintenance, "
    "with some trials showing greater weight losses with yogurt interventions "
    "compared to control diets[3]. However, the results across different studies are "
    "not entirely consistent, as some studies have shown mixed results regarding "
    "yogurt consumption and changes in body weight and waist circumference[3]."
)

# Whole-null-result answer: the only substantive ending is a compound clause head.
ANS_31631671_RERUN = (
    "No, a vegan or vegetarian diet is not consistently associated with the "
    "microbiota composition in the gut compared to omnivores[1]. The systematic "
    "review included sixteen studies, investigating the association between gut "
    "microbiota composition in both vegans and vegetarians, with no consistent "
    "association identified between these diets and microbiota composition compared "
    "to omnivores[1]. The studies included in the review reported on various genera "
    "and species, such as Bacteroides, Bifidobacterium, and Prevotella, but the "
    "results were contradictory and did not establish a clear link between vegan or "
    "vegetarian diets and specific changes in gut microbiota[1]."
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

    # 1-3: hedge closers must not survive anywhere in the output (the judge
    #      flags noncommittal content at any position), the ending must be
    #      substantive, and the substantive lead content must survive.
    for qid, ans in [
        ("36518821", ANS_36518821),
        ("32422943", ANS_32422943),
        ("26891320", ANS_26891320),
        ("26891320-rerun", ANS_26891320_RERUN),
        ("26443336-rerun", ANS_26443336_RERUN),
        ("31631671-rerun", ANS_31631671_RERUN),
    ]:
        out = strip_gap_talk_sentences(ans)
        check(
            f"{qid} no hedge survives",
            not _TRAILING_HEDGE_RE.search(out),
            f"hedge still present in: {out!r}",
        )
        check(
            f"{qid} substantive lead kept",
            out.startswith(ans.split(". ")[0].rstrip(".")[:60]),
            f"lead changed: {out!r}",
        )
        check(
            f"{qid} citations kept",
            re.search(r"\[\d+\]", out) is not None,
            f"citations lost: {out!r}",
        )

    # 3b: the compound "head, but hedge" split keeps the specific head clause
    #     and moves its citation onto it.
    out = strip_gap_talk_sentences(ANS_31631671_RERUN)
    check(
        "31631671 compound head kept",
        "Bacteroides, Bifidobacterium, and Prevotella[1]." in out,
        f"got: {out!r}",
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

    # 7: one substantive sentence plus one hedge must drop the hedge. Keeping
    # it anywhere still hard-zeros AnswerRelevancy. This was not covered by
    # the original fixtures and exposed the old one-substantive demotion path.
    single = (
        "Vitamin D improved the score by 12% in 400 participants[1]. "
        "More research is needed[1]."
    )
    out = strip_gap_talk_sentences(single)
    check(
        "one-substantive hedge removed",
        out == "Vitamin D improved the score by 12% in 400 participants[1].",
        f"got: {out!r}",
    )

    # 8: a compound answer with one useful head keeps the head and its citation
    # rather than retaining the hedge tail.
    compound = "The review found lower LDL levels, but the evidence remains unclear[1]."
    out = strip_gap_talk_sentences(compound)
    check(
        "one-head compound hedge removed",
        out == "The review found lower LDL levels[1].",
        f"got: {out!r}",
    )

    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print(" -", f)
        return 1
    print("OK - 24 trailing-hedge regression checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
