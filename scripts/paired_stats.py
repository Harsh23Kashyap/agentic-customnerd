#!/usr/bin/env python3
"""Paired statistics for a rule-vs-original (OFF) benchmark sheet.

Input: the "Paired metrics" sheet of an .xlsx (needs openpyxl), or a CSV with
the same column names. Per metric (seconds, LLM calls, tokens) it reports,
for all rows and for row subsets given with --split:

  n, totals, ratio of totals, geometric-mean per-question ratio (OFF/rule),
  exact 95% sign-test-free bootstrap CI of that geometric mean (seeded),
  exact two-sided Wilcoxon signed-rank p-value on the paired differences,
  and the count of questions where OFF > rule.

Direction: every ratio is OFF / rule (values above 1 mean the original flow used
more). "ratio_of_totals" is sum(OFF)/sum(rule) and is dominated by the heaviest
questions; "geo_mean_ratio" weights every question equally. Report which one
you mean. Wilcoxon is two-sided, zero differences are dropped, tied
|differences| get average ranks, and the p-value is exact for the tied ranks.
Standard library only for the maths. Nothing is called over the network.
Cohort mixing is the caller's problem: use --split to report subsets
(for example rows 0-5 and 6-19) next to the overall result.

Usage:
  python paired_stats.py results.xlsx [--sheet "Paired metrics"] \
      [--split 0-5 --split 6-19] [--boot 10000] [--seed 0] [--json]
"""
import argparse
import csv
import itertools
import json
import math
import random
import sys

METRICS = {
    "seconds": ("Rule seconds", "OFF seconds"),
    "llm_calls": ("Rule LLM calls", "OFF LLM calls"),
    "tokens": ("Rule tokens", "OFF tokens"),
}


def load_rows(path, sheet):
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    import openpyxl

    ws = openpyxl.load_workbook(path, data_only=False)[sheet]
    it = ws.iter_rows(values_only=True)
    header = next(it)
    return [dict(zip(header, r)) for r in it if r and r[0] is not None]


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def wilcoxon_exact(diffs):
    """Exact two-sided Wilcoxon signed-rank p-value (zeros dropped, ties by average rank)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = sum(ranks)
    w = min(w_plus, total - w_plus)
    # Exact null via DP over doubled ranks (handles ties); n <= ~60 is fine.
    dr = [int(round(2 * r)) for r in ranks]
    top = sum(dr)
    counts = [0] * (top + 1)
    counts[0] = 1
    for r in dr:
        for s in range(top, r - 1, -1):
            counts[s] += counts[s - r]
    limit = int(math.floor(2 * w + 1e-9))
    tail = sum(counts[: limit + 1])
    return min(1.0, 2.0 * tail / (2 ** n))


def geo_mean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def boot_ci(ratios, boot, seed):
    rng = random.Random(seed)
    n = len(ratios)
    logs = [math.log(r) for r in ratios]
    means = sorted(sum(rng.choice(logs) for _ in range(n)) / n for _ in range(boot))
    lo = means[int(0.025 * boot)]
    hi = means[min(boot - 1, int(0.975 * boot))]
    return math.exp(lo), math.exp(hi)


def analyse(rows, rule_col, off_col, boot, seed):
    pairs, skipped = [], []
    for r in rows:
        a, b = num(r.get(rule_col)), num(r.get(off_col))
        if a is None or b is None or a <= 0 or b <= 0:
            skipped.append(r.get("Row"))
            continue
        pairs.append((a, b))
    if len(pairs) < 2:
        return {"n": len(pairs), "skipped_rows": skipped, "note": "need at least 2 valid pairs"}
    rule = [a for a, _ in pairs]
    off = [b for _, b in pairs]
    ratios = [b / a for a, b in pairs]
    lo, hi = boot_ci(ratios, boot, seed)
    return {
        "n": len(pairs),
        "skipped_rows": skipped,
        "rule_total": sum(rule),
        "off_total": sum(off),
        "ratio_of_totals": sum(off) / sum(rule),
        "geo_mean_ratio": geo_mean(ratios),
        "geo_mean_ratio_ci95_bootstrap": [lo, hi],
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "off_greater_count": sum(1 for a, b in pairs if b > a),
        "wilcoxon_exact_p_two_sided": wilcoxon_exact([b - a for a, b in pairs]),
    }


def parse_split(s):
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--sheet", default="Paired metrics")
    ap.add_argument("--split", action="append", default=[], help="row range like 0-5 (repeatable)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rows = load_rows(args.path, args.sheet)
    groups = {"all": rows}
    for s in args.split:
        lo, hi = parse_split(s)
        groups[f"rows {lo}-{hi}"] = [r for r in rows if num(r.get("Row")) is not None and lo <= num(r["Row"]) <= hi]
    out = {g: {m: analyse(rs, *cols, args.boot, args.seed) for m, cols in METRICS.items()} for g, rs in groups.items()}
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        print()
        return out
    for g, ms in out.items():
        print(f"== {g}")
        for m, st in ms.items():
            if "ratio_of_totals" not in st:
                print(f"  {m}: n={st['n']} ({st.get('note')})")
                continue
            lo, hi = st["geo_mean_ratio_ci95_bootstrap"]
            print(
                f"  {m}: n={st['n']} ratio_of_totals={st['ratio_of_totals']:.2f}x "
                f"geo_mean={st['geo_mean_ratio']:.2f}x (95% boot CI {lo:.2f}-{hi:.2f}) "
                f"range {st['min_ratio']:.2f}-{st['max_ratio']:.2f} "
                f"OFF>rule {st['off_greater_count']}/{st['n']} wilcoxon_p={st['wilcoxon_exact_p_two_sided']:.2g}"
            )
    return out


if __name__ == "__main__":
    main()
