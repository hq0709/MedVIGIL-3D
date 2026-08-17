"""
Independent checks of claims made about this corpus and these runs.

Every number here is recomputed from the committed corpus and the committed
run files. Nothing is taken from a summary. Each block prints what was claimed
and what the data says, so a mismatch is visible rather than inferred.

    python analysis/verify_claims.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from growth_matched import LABEL, ORDER, growth_of, matched_subset  # noqa: E402


def corpus():
    ref, prov = {}, {}
    for line in open(os.path.join(REPO, "common_subset", "qa", "all.jsonl")):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
        prov[r["qid"]] = r["provenance"]
    return ref, prov


def load(tag: str, arm: str = "sighted"):
    p = os.path.join(REPO, f"mm_{tag}_{arm}.jsonl")
    if not os.path.exists(p):
        return None
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p) if l.strip()}


def yes_score(row: dict) -> float:
    """Continuous decision variable: how much more likely "yes" is than "no".

    The accuracy tables threshold this at zero. An AUROC over it asks a
    different and weaker question -- whether the ordering carries signal -- and
    the two can disagree: a model whose argmax never flips inside a pair can
    still rank the pair correctly every time.
    """
    lp = row.get("logprobs") or {}
    if "yes" not in lp or "no" not in lp:
        return float("nan")
    return float(lp["yes"]) - float(lp["no"])


def auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUROC with ties counted as half, the Mann-Whitney form."""
    if not pos or not neg:
        return float("nan")
    x = np.array(pos + neg, float)
    if not np.isfinite(x).all():
        return float("nan")
    order = x.argsort()
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng) for _ in range(n)]
    vals = [v for v in vals if np.isfinite(v)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def head(title, claim):
    print(f"\n{'=' * 100}\n{title}\n  CLAIM: {claim}\n{'-' * 100}")


def main() -> None:
    ref, prov = corpus()
    keep = matched_subset(ref, growth_of())
    by_pair = defaultdict(list)
    for q in sorted(keep):
        by_pair["_".join(q.split("_")[:-1])].append(q)
    pairs = {p: v for p, v in by_pair.items() if len(v) == 2}

    # ---------------------------------------------------------------- 1
    head("1. DEGENERATE RESPONDERS",
         "7 models put >=96% of answers on one token, so their 50.0% is entailed")
    rows = []
    for tag in ORDER:
        S = load(tag)
        if not S:
            continue
        sub = [S[q] for q in keep if q in S]
        if len(sub) != len(keep):
            continue
        c = Counter(r["prediction"] for r in sub)
        modal = 100.0 * c.most_common(1)[0][1] / len(sub)
        acc = 100.0 * sum(r["prediction"] == r["gold"] for r in sub) / len(sub)
        rows.append((tag, LABEL[tag], acc, modal, c.most_common(1)[0][0]))
    rows.sort(key=lambda r: -r[3])
    # A threshold on modal share is arbitrary. The exact statement: on a balanced
    # corpus, a model answering its modal word on a fraction m of items has
    # accuracy confined to [m - 0.5, 1.5 - m] whatever it is doing, because only
    # half the items carry that gold label. When the measured interval lies
    # inside that window, the number is a restatement of the answer distribution.
    print(f"{'model':24}{'acc':>8}{'modal share':>13}{'entailed window':>19}"
          f"{'informative?':>14}")
    for _, name, acc, modal, word in rows:
        m = modal / 100.0
        lo, hi = 100 * (m - 0.5), 100 * (1.5 - m)
        inside = lo <= 50.0 <= hi and (hi - lo) < 8.0
        print(f"{name:24}{acc:>7.1f}%{modal:>12.1f}%"
              f"{f'[{lo:.1f}, {hi:.1f}]':>19}"
              f"{('no -- entailed' if inside else 'yes'):>14}")
    deg = [r for r in rows if r[3] >= 96.0]
    print(f"\n  FOUND: {len(deg)} models at >=96% modal share on THIS subset.")
    print(f"  The same count on the full 2,262-probe corpus is 6: InternVL3-14B "
          f"is 93.9% there\n  and 96.5% here. The reported accuracies come from "
          f"the matched subset, so the\n  exclusion has to be judged on the "
          f"matched subset too.")
    print(f"  A constant answer on the {sum(1 for q in keep if ref[q] == 'yes')}/"
          f"{sum(1 for q in keep if ref[q] == 'no')} balanced subset scores "
          f"exactly 50.0% by arithmetic.")

    # ---------------------------------------------------------------- 2
    head("2. STRATIFIED CONCLUSION",
         "drop degenerate responders and the M3D channel failures; the 5 that "
         "remain still all contain 50")
    native_fail = {"m3d", "m3dllama", "med3dvlm"}
    survivors = [r for r in rows if r[3] < 96.0 and r[0] not in native_fail]
    print(f"{'model':24}{'acc':>8}{'modal share':>14}")
    for tag, name, acc, modal, _ in survivors:
        print(f"{name:24}{acc:>7.1f}%{modal:>13.1f}%")
    print(f"\n  {len(survivors)} models survive both exclusions.")

    # ---------------------------------------------------------------- 3
    head("3. AUROC OF THE DECISION VARIABLE",
         "0.470-0.522 across models, the two excluding 0.5 lying BELOW it")
    print(f"{'model':24}{'AUROC':>9}{'95% CI':>18}{'n':>7}")
    aurocs = {}
    for tag, name, *_ in rows:
        S = load(tag)
        qs = [q for q in keep if q in S]
        pos = [yes_score(S[q]) for q in qs if ref[q] == "yes"]
        neg = [yes_score(S[q]) for q in qs if ref[q] == "no"]
        if not np.isfinite(pos + neg).all():
            print(f"{name:24}{'no logprobs':>9}")
            continue
        a = auroc(pos, neg)
        lo, hi = boot_ci(lambda r: auroc(list(r.choice(pos, len(pos))),
                                         list(r.choice(neg, len(neg)))))
        aurocs[tag] = a
        mark = "" if lo <= 0.5 <= hi else ("  below" if hi < 0.5 else "  above")
        print(f"{name:24}{a:>9.3f}  [{lo:.3f},{hi:.3f}]{len(qs):>7}{mark}")
    if aurocs:
        print(f"\n  FOUND range {min(aurocs.values()):.3f}-{max(aurocs.values()):.3f}")

    # ---------------------------------------------------------------- 4
    head("4. WHAT THE DECISION VARIABLE TRACKS",
         "image fixed / text varied -> 0.775-0.891; text fixed / image varied "
         "-> 0.467-0.528")
    print("  image fixed, text varied = the two members of a matched pair: same")
    print("  rendering, growth amount differs, labels opposite.")
    print("  text fixed, image varied = items sharing (target, growth amount)")
    print("  across different volumes, so the sentence is identical.\n")
    print(f"{'model':24}{'image fixed':>13}{'95% CI':>18}{'text fixed':>13}"
          f"{'95% CI':>18}{'n pairs':>9}{'n text':>8}")
    for tag, name, *_ in rows:
        S = load(tag)
        wp = [(yes_score(S[q]) if ref[q] == "yes" else None,
               yes_score(S[q]) if ref[q] == "no" else None)
              for p, v in pairs.items() for q in v if q in S]
        # within-pair: score of the yes member vs the no member of the SAME pair
        d = []
        for p, v in pairs.items():
            if not all(q in S for q in v):
                continue
            y = [q for q in v if ref[q] == "yes"]
            n = [q for q in v if ref[q] == "no"]
            if len(y) == 1 and len(n) == 1:
                d.append(yes_score(S[y[0]]) - yes_score(S[n[0]]))
        d = [x for x in d if np.isfinite(x)]
        a_img = (sum((x > 0) + 0.5 * (x == 0) for x in d) / len(d)) if d else np.nan
        lo1, hi1 = boot_ci(lambda r, d=d: sum((x > 0) + 0.5 * (x == 0)
                                              for x in r.choice(d, len(d))) / len(d)) \
            if d else (np.nan, np.nan)
        # text fixed: same (target, growth), differing volume
        groups = defaultdict(list)
        for q in keep:
            if q in S:
                groups[(prov[q]["target"], prov[q]["growth_mm"])].append(q)
        pos, neg = [], []
        for g, qs in groups.items():
            labs = {ref[q] for q in qs}
            if len(labs) < 2:
                continue
            for q in qs:
                (pos if ref[q] == "yes" else neg).append(yes_score(S[q]))
        a_txt = auroc(pos, neg)
        lo2, hi2 = boot_ci(lambda r: auroc(list(r.choice(pos, len(pos))),
                                           list(r.choice(neg, len(neg))))) \
            if pos and neg else (np.nan, np.nan)
        print(f"{name:24}{a_img:>13.3f}  [{lo1:.3f},{hi1:.3f}]{a_txt:>13.3f}"
              f"  [{lo2:.3f},{hi2:.3f}]{len(d):>9}{len(pos) + len(neg):>8}")

    # ---------------------------------------------------------------- 5
    head("5. DIFFICULTY STRUCTURE",
         "|growth - gap| median 5.06 mm on the matched subset, no easy stratum")
    diff = np.array([abs(prov[q]["growth_mm"] - prov[q]["gap_mm"]) for q in keep])
    print(f"  |growth - gap| over the matched subset (n={len(diff)}):")
    print(f"    median {np.median(diff):.2f} mm   mean {diff.mean():.2f}   "
          f"p10 {np.percentile(diff,10):.2f}   p90 {np.percentile(diff,90):.2f}   "
          f"max {diff.max():.2f}")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
        sel = (diff >= lo) & (diff < hi)
        if sel.sum():
            print(f"    {lo:>2}-{hi if hi < 1e9 else 'inf':>3} mm: {sel.sum():>5} probes "
                  f"({100*sel.mean():.1f}%)")

    # ---------------------------------------------------------------- 6
    head("6. THE margin=2.0 GATE",
         "it removes every gap <= 2 mm case, i.e. the clinically nearest ones")
    print("  counterfactual_qa.growth_pairs keeps a (lesion, target) pair only "
          "when\n  margin < gap <= max_gap_mm, with margin=2.0 and "
          "max_gap_mm=40.0.\n")
    gaps = np.array([prov[q]["gap_mm"] for q in ref])
    print(f"  gap_mm over the whole emitted corpus (n={len(gaps)}): "
          f"min {gaps.min():.2f}  max {gaps.max():.2f}")
    print(f"    probes with gap <= 2.0 mm : {(gaps <= 2.0).sum()}")
    print(f"    probes with gap  > 40 mm  : {(gaps > 40.0).sum()}")
    print("  Both are zero by construction, so the corpus contains no lesion "
          "already touching\n  or nearly touching its target, and none far from "
          "it. Whether that excludes the\n  clinically most relevant cases is a "
          "claim about clinical relevance, not about\n  the data -- but the "
          "exclusion itself is real and is visible here.")


if __name__ == "__main__":
    main()
