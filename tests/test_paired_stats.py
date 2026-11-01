import csv
import itertools
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import paired_stats as ps  # noqa: E402

try:
    import openpyxl  # noqa: F401
    HAVE_XLSX = True
except ImportError:  # pragma: no cover
    HAVE_XLSX = False


def rows_from(pairs, metric=("Rule seconds", "OFF seconds")):
    return [{"Row": i, metric[0]: a, metric[1]: b} for i, (a, b) in enumerate(pairs)]


def brute_force_wilcoxon(d):
    """Independent oracle: enumerate all sign assignments (average ranks, zeros dropped)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ab = [abs(x) for x in d]
    srt = sorted(ab)
    ranks = []
    for x in ab:
        idx = [i for i, v in enumerate(srt) if v == x]
        ranks.append((idx[0] + idx[-1]) / 2 + 1)
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    tot = sum(ranks)
    w = min(wp, tot - wp)
    hit = 0
    for signs in itertools.product((0, 1), repeat=n):
        s = sum(r for r, g in zip(ranks, signs) if g)
        if min(s, tot - s) <= w + 1e-9:
            hit += 1
    return hit / 2**n


class Wilcoxon(unittest.TestCase):
    def test_matches_brute_force_with_ties_and_zeros(self):
        import random

        rng = random.Random(1)
        for _ in range(150):
            d = [rng.choice([-3, -2, -1, 0, 0, 1, 1, 2, 3, 4]) for _ in range(rng.randint(1, 11))]
            self.assertAlmostEqual(ps.wilcoxon_exact(d), brute_force_wilcoxon(d), places=12, msg=str(d))

    def test_matches_scipy_when_available(self):
        try:
            from scipy import stats
        except ImportError:
            self.skipTest("scipy not installed")
        import random

        rng = random.Random(2)
        for _ in range(100):
            d = [rng.uniform(-5, 8) for _ in range(rng.randint(2, 16))]
            ref = stats.wilcoxon(d, method="exact", zero_method="wilcox", alternative="two-sided").pvalue
            self.assertAlmostEqual(ps.wilcoxon_exact(d), ref, places=9)

    def test_all_positive_n20_is_two_over_2n(self):
        self.assertAlmostEqual(ps.wilcoxon_exact(list(range(1, 21))), 2 / 2**20)

    def test_symmetric_is_one(self):
        self.assertEqual(ps.wilcoxon_exact([1, -1, 2, -2]), 1.0)

    def test_zeros_dropped(self):
        self.assertEqual(ps.wilcoxon_exact([0, 0, 0]), 1.0)

    def test_known_small_value(self):
        # diffs 1,2,3 all positive: W-=0, two-sided p = 2/8
        self.assertAlmostEqual(ps.wilcoxon_exact([1, 2, 3]), 0.25)

    def test_ties_use_average_ranks(self):
        p = ps.wilcoxon_exact([1, 1, -1, 2])
        self.assertTrue(0 < p <= 1)

    def test_sign_flip_invariant(self):
        d = [3, 1, -2, 5, 4]
        self.assertAlmostEqual(ps.wilcoxon_exact(d), ps.wilcoxon_exact([-x for x in d]))


class Analyse(unittest.TestCase):
    def test_ratio_of_totals_and_geo_mean(self):
        a = ps.analyse(rows_from([(1, 2), (2, 8)]), "Rule seconds", "OFF seconds", 200, 0)
        self.assertAlmostEqual(a["ratio_of_totals"], 10 / 3)
        self.assertAlmostEqual(a["geo_mean_ratio"], math.sqrt(2 * 4))
        self.assertEqual(a["off_greater_count"], 2)

    def test_skips_missing_and_nonpositive(self):
        rows = rows_from([(1, 2), (None, 5), (0, 3), (2, 4)])
        a = ps.analyse(rows, "Rule seconds", "OFF seconds", 100, 0)
        self.assertEqual(a["n"], 2)
        self.assertEqual(sorted(a["skipped_rows"]), [1, 2])

    def test_too_few_pairs_reports_note(self):
        a = ps.analyse(rows_from([(1, 2)]), "Rule seconds", "OFF seconds", 100, 0)
        self.assertIn("note", a)

    def test_seeded_bootstrap_is_reproducible(self):
        rs = rows_from([(1, 2), (2, 5), (3, 5), (1, 3)])
        x = ps.analyse(rs, "Rule seconds", "OFF seconds", 500, 7)
        y = ps.analyse(rs, "Rule seconds", "OFF seconds", 500, 7)
        self.assertEqual(x, y)

    def test_ci_contains_point_estimate_for_constant_ratio(self):
        a = ps.analyse(rows_from([(1, 3), (2, 6), (5, 15)]), "Rule seconds", "OFF seconds", 500, 0)
        lo, hi = a["geo_mean_ratio_ci95_bootstrap"]
        self.assertAlmostEqual(lo, 3.0)
        self.assertAlmostEqual(hi, 3.0)

    def test_equal_pair_not_counted_as_off_greater(self):
        a = ps.analyse(rows_from([(5, 5), (5, 9)]), "Rule seconds", "OFF seconds", 100, 0)
        self.assertEqual(a["off_greater_count"], 1)

    def test_off_cheaper_counted(self):
        a = ps.analyse(rows_from([(5, 1), (5, 9), (5, 9)]), "Rule seconds", "OFF seconds", 100, 0)
        self.assertEqual(a["off_greater_count"], 2)


class CliIo(unittest.TestCase):
    def _csv(self, rows):
        fd, p = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Row", "Rule seconds", "OFF seconds", "Rule LLM calls", "OFF LLM calls", "Rule tokens", "OFF tokens"])
            w.writeheader()
            w.writerows(rows)
        self.addCleanup(os.remove, p)
        return p

    def _data(self):
        return [
            {"Row": i, "Rule seconds": 10, "OFF seconds": 30 + i, "Rule LLM calls": 5, "OFF LLM calls": 20,
             "Rule tokens": 100, "OFF tokens": 1000 + i}
            for i in range(8)
        ]

    def test_csv_split_groups(self):
        out = ps.main([self._csv(self._data()), "--split", "0-2", "--split", "3-7", "--json", "--boot", "100"])
        self.assertEqual(out["rows 0-2"]["seconds"]["n"], 3)
        self.assertEqual(out["rows 3-7"]["seconds"]["n"], 5)
        self.assertEqual(out["all"]["tokens"]["n"], 8)

    @unittest.skipUnless(HAVE_XLSX, "openpyxl not installed")
    def test_xlsx_roundtrip_matches_csv(self):
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Paired metrics"
        hdr = ["Row", "Rule seconds", "OFF seconds", "Rule LLM calls", "OFF LLM calls", "Rule tokens", "OFF tokens"]
        ws.append(hdr)
        for r in self._data():
            ws.append([r[h] for h in hdr])
        fd, p = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        self.addCleanup(os.remove, p)
        wb.save(p)
        a = ps.main([p, "--json", "--boot", "100"])
        b = ps.main([self._csv(self._data()), "--json", "--boot", "100"])
        self.assertEqual(a, b)

    def test_text_output_runs(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ps.main([self._csv(self._data()), "--boot", "100"])
        self.assertIn("ratio_of_totals", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
