"""Regression tests for the disclaimer / gap-talk strip in claim_filter.

These tests run without pytest. Execute with:
    python agent/tests/test_no_disclaimer.py

The end-to-end numbers (20Q run, 6/20 → 0/20 with disclaimer, RAGAS proxy
Rel +0.033, Faith +0.037) live in the eval report. These tests pin the
piecewise behavior of the strip so a future refactor cannot silently regress
RAGAS Rel=0 on DietNerd-style medical answers.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_claim_filter():
    """Load claim_filter without triggering the full agent import chain.

    The module imports `agent.state.AgentState` at the top, which loads
    orchestrator → openai_executions. We stub those so the test runs in a
    plain venv without the legacy backend on the path.
    """
    src_path = REPO_ROOT / "agent" / "tools" / "claim_filter.py"
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    with open(src_path) as f:
        src = f.read()

    # Stub the agent.state import.
    fake_state = types.ModuleType("agent.state")

    class _Stub:
        pass

    fake_state.AgentState = _Stub
    fake_agent = types.ModuleType("agent")
    fake_agent.state = fake_state
    sys.modules.setdefault("agent", fake_agent)
    sys.modules.setdefault("agent.state", fake_state)

    namespace: dict = {"__name__": "claim_filter_under_test"}
    exec(compile(src, str(src_path), "exec"), namespace)
    return namespace["__name__"] and namespace or namespace


cf = _load_claim_filter()
strip_disclaimer = cf["strip_disclaimer"]
strip_gap_talk_sentences = cf["strip_gap_talk_sentences"]
verify_citation_attribution = cf["verify_citation_attribution"]


# --- Tests -----------------------------------------------------------------


class TestStripDisclaimer(unittest.TestCase):
    def test_strips_trailing_block(self):
        ans = (
            "Folic acid improves memory in older adults with deficiency [1].\n\n"
            "**Disclaimer**: This is informational only and not a substitute "
            "for professional medical advice."
        )
        out = strip_disclaimer(ans)
        self.assertNotIn("Disclaimer", out)
        self.assertNotIn("informational", out.lower())
        self.assertIn("folic acid", out.lower())

    def test_strips_inline_sentence(self):
        ans = (
            "The trial enrolled 200 participants. The result was significant. "
            "Always consult a qualified healthcare provider before changing your regimen."
        )
        out = strip_disclaimer(ans)
        self.assertNotIn("consult", out.lower())
        self.assertIn("200 participants", out)

    def test_preserves_evidence_backed_closer(self):
        # "Further trials are needed" without study numbers should drop,
        # but a sentence that carries n=, p<, etc. should survive.
        ans = (
            "The meta-analysis found a pooled effect of SMD = -0.42 (95% CI -0.61 to -0.23) "
            "across 14 RCTs (n=2,341 participants). Further randomized controlled trials are needed."
        )
        out = strip_gap_talk_sentences(ans)
        self.assertIn("SMD = -0.42", out)
        self.assertIn("2,341", out)
        # The bare "Further RCTs are needed" with no numbers in it should drop.
        self.assertNotIn("Further randomized controlled trials are needed", out)

    def test_keeps_evidence_with_inline_further_research(self):
        # "Further research is needed" at the end of a sentence carrying a
        # p-value and n= should NOT drop (study n is structural for Faith).
        ans = (
            "The trial reported a significant effect (p<0.01, n=412). "
            "Further research is needed to confirm the long-term outcomes."
        )
        out = strip_gap_talk_sentences(ans)
        self.assertIn("p<0.01", out)
        self.assertIn("n=412", out)

    def test_no_disclaimer_passthrough(self):
        ans = "Vitamin D supplementation reduces fracture risk in adults over 65 [1]."
        out = strip_disclaimer(ans)
        self.assertEqual(out, ans)

    def test_empty_input(self):
        self.assertEqual(strip_disclaimer(""), "")
        self.assertEqual(strip_disclaimer(None or ""), "")


class TestVerifyCitationAttribution(unittest.TestCase):
    def setUp(self):
        self.cites = [
            {
                "title": "Folic acid and memory in older adults",
                "abstract": (
                    "A randomized controlled trial of folic acid supplementation "
                    "showed improved memory scores in 200 participants with deficiency."
                ),
            },
            {
                "title": "Mediterranean diet and cardiovascular outcomes",
                "abstract": (
                    "A 5-year cohort study of plant-predominant eating patterns "
                    "showed reduced cardiovascular events in 12,000 participants."
                ),
            },
        ]

    def test_well_attributed_marker_passes(self):
        ans = (
            "Folic acid supplementation improved memory in 200 participants with "
            "deficiency [1]."
        )
        result = verify_citation_attribution(ans, self.cites)
        self.assertEqual(result["attributed_count"], 1)
        self.assertEqual(result["unverified"], [])
        self.assertEqual(result["missing_citation"], [])

    def test_low_overlap_marker_flagged(self):
        # Sentence about a totally different topic citing article 1 — should flag.
        ans = "The spacecraft reentry trajectory required a heat shield rated to 3000 Kelvin [1]."
        result = verify_citation_attribution(ans, self.cites)
        self.assertEqual(result["attributed_count"], 0)
        self.assertEqual(len(result["unverified"]), 1)
        self.assertEqual(result["unverified"][0]["marker"], "[1]")
        self.assertLess(result["unverified"][0]["overlap"], 0.10)

    def test_out_of_range_marker_missing(self):
        ans = "Some claim [3]."  # index 3 doesn't exist
        result = verify_citation_attribution(ans, self.cites)
        self.assertEqual(result["attributed_count"], 0)
        self.assertEqual(len(result["missing_citation"]), 1)
        self.assertEqual(result["missing_citation"][0]["marker"], "[3]")

    def test_multiple_markers_in_one_sentence(self):
        ans = (
            "Folic acid improved memory in 200 participants [1] and plant-predominant "
            "eating patterns reduced cardiovascular events in 12,000 participants [2]."
        )
        result = verify_citation_attribution(ans, self.cites)
        self.assertEqual(result["attributed_count"], 2)
        self.assertEqual(result["unverified"], [])

    def test_no_markers_returns_zero(self):
        ans = "No citations here, just a plain claim about nutrition."
        result = verify_citation_attribution(ans, self.cites)
        self.assertEqual(result["cited_sentence_count"], 0)
        self.assertEqual(result["total_citation_markers"], 0)
        self.assertEqual(result["attributed_count"], 0)


# --- Standalone runner -----------------------------------------------------

if __name__ == "__main__":
    failures = 0
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll tests passed.")
        sys.exit(0)
    sys.exit(1)
