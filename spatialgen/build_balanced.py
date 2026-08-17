"""
Regenerate the counterfactual corpus with the growth-magnitude confound removed
at the source, rather than selected away afterwards.

Why the original construction leaks
-----------------------------------
counterfactual_qa.growth_pairs picks growth amounts *relative to the gap it just
measured*:

    below = gap - margin - U(0, 0.3*gap)     -> answer "no"
    above = gap + margin + U(0, 5)           -> answer "yes"

Both track the gap and "yes" is always the larger, so across the corpus the
marginal distribution of growth differs by label (median 26.0 mm vs 15.3 mm) and
a threshold on the number alone reaches 69.2 %. Within a pair that ordering is
the design -- it is what pins pair-level chance at 50 % -- but the marginal was
never controlled.

The fix: invert the sampling direction
--------------------------------------
Choose the growth amount FIRST, from a fixed global grid, then find geometry
whose gap straddles it. A pair with gap g contributes its "no" question at some
grid value below g and its "yes" question at some grid value above g. If, for
every grid value G, the number of pairs using G as their "yes" equals the number
using it as their "no", then P(growth | yes) = P(growth | no) exactly -- while
matched pairs are preserved intact.

This requires no new geometry. Every (lesion, target, gap_mm) triple already
exists in the provenance of the original corpus, and gap_mm is the only quantity
needed to answer any growth query: contact after growing by G occurs iff
gap <= G. The corpus is re-derived, not re-measured.
"""
from __future__ import annotations

import os
from pathlib import Path

import argparse
import glob
import json
import random
from collections import Counter, defaultdict


def load_pairs(root: str) -> dict:
    """(pair_id) -> dict with gap, target, organ, volume."""
    out = {}
    for f in glob.glob(f"{root}/cfqa_*/qa/*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("kind") != "growth_contact":
                continue
            p = r["provenance"]
            if p.get("gap_mm") is None or not r.get("pair_id"):
                continue
            out[r["pair_id"]] = {
                "gap": float(p["gap_mm"]), "target": p["target"],
                "organ": r.get("organ") or f.split("/")[-3].replace("cfqa_", ""),
                "volume": "_".join(r["qid"].split("_")[:2]),
            }
    return out


def assign(pairs: dict, grid: list[float], margin: float, cap: int, seed: int):
    """Assign each kept pair ONE question, so that every grid value carries
    equal numbers of yes and no.

    Singletons rather than matched pairs, deliberately. A pair emitting both its
    "no" (at a grid value below its gap) and its "yes" (above) couples two grid
    values together, and per-G balancing then has to trim along both edges at
    once; on this gap distribution that cascade removes the whole corpus. The two
    properties are in tension, so we separate them: accuracy and the intervention
    are measured here, where the marginal cue is provably absent, and pair
    consistency stays on the original corpus, where pairs are intact.

    Candidates nearest the grid value are preferred, making each question as
    hard as the grid allows -- the growth amount sits just under or just over the
    true gap.
    """
    rng = random.Random(seed)
    used: set = set()
    out: dict = {}
    # widest grid values first: they have the scarcest candidate pools
    order = sorted(grid, key=lambda G: -abs(G - (grid[0] + grid[-1]) / 2))
    for G in order:
        yes = sorted((d["gap"], p) for p, d in pairs.items()
                     if p not in used and d["gap"] <= G - margin)
        no = sorted((d["gap"], p) for p, d in pairs.items()
                    if p not in used and d["gap"] >= G + margin)
        yes.sort(key=lambda x: -x[0])       # gap closest to G from below
        no.sort(key=lambda x: x[0])         # gap closest to G from above
        k = min(len(yes), len(no), cap)
        for i in range(k):
            out[yes[i][1]] = (G, "yes")
            out[no[i][1]] = (G, "no")
            used.add(yes[i][1])
            used.add(no[i][1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("MEDVIGIL3D_ROOT", str(Path(__file__).resolve().parent.parent)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = load_pairs(args.root)
    grid = [round(x, 1) for x in
            [args.step * i for i in range(1, int(45 / args.step) + 1)]]
    asg = assign(pairs, grid, args.margin, args.cap, args.seed)
    print(f"{len(pairs)} pairs available -> {len(asg)} questions, grid {grid}")

    n = 0
    lab = Counter()
    byl = defaultdict(Counter)
    with open(args.out, "w") as f:
        for pid, (G, ans) in sorted(asg.items()):
            d = pairs[pid]
            assert (d["gap"] <= G) == (ans == "yes"), "label/geometry mismatch"
            tgt = d["target"].replace("_", " ")
            f.write(json.dumps({
                "qid": f"{pid}_b{G:g}", "kind": "growth_contact",
                "question": (f"If this lesion grew by {G:g} mm in every "
                             f"direction, would it contact the {tgt}? "
                             f"Answer yes or no."),
                "answer": ans, "choices": ["no", "yes"], "pair_id": pid,
                "organ": d["organ"],
                "provenance": {"target": d["target"], "gap_mm": d["gap"],
                               "growth_mm": G,
                               "rule": "surface distance after isotropic growth"},
            }) + "\n")
            lab[ans] += 1
            byl[ans][G] += 1
            n += 1

    ok = all(byl["no"][G] == byl["yes"][G] for G in grid)
    items = [(G, a) for a in ("yes", "no") for G, c in byl[a].items()
             for _ in range(c)]
    best = max((sum((G > t) == (a == "yes") for G, a in items) / len(items)
                for t in grid), default=0)
    print(f"wrote {n} questions ({dict(lab)})")
    print(f"per-G label balance exact: {ok}")
    print(f"growth-threshold ceiling: {100*best:.1f}%  (50.0% = cue absent)")
    for G in grid:
        if byl["no"][G] or byl["yes"][G]:
            print(f"   G={G:5g}mm   no {byl['no'][G]:5d}   yes {byl['yes'][G]:5d}")


if __name__ == "__main__":
    main()
