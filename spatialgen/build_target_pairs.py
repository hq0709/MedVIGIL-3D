"""
Target-contrast pairs: matched probes that a numeric comparison cannot separate.

Why the original pairing had to be replaced
-------------------------------------------
The growth-contrast pair holds the lesion and target fixed and varies the growth
amount, so the two members differ by one number and the larger is always the
"yes". Consistency -- answering the two differently -- is then achievable by
comparing two integers in the prompt, and measurement confirmed it: across every
model, accuracy within consistent pairs equals the rate of assigning "yes" to the
larger amount, to the decimal. The metric was real but it was not measuring
grounding.

The construction here removes that route. Both members state the **same growth
amount** on the **same lesion**, and differ only in which structure is named --
one whose gap is under that amount, one whose gap is over it. The numbers are
identical, so ordering them cannot produce a differentiated answer; the model
must distinguish two named structures with respect to one lesion.

What this does and does not establish
-------------------------------------
It removes the numeric shortcut, not every shortcut: a model with strong
anatomical priors could sometimes rank two structures by typical distance without
consulting the scan. That is exactly what the blind arm measures, and it is why
these probes are run in both arms rather than reported on their own. The claim
they can support is narrower than the original one and actually testable:
consistency here requires distinguishing two structures, and the sighted-blind
difference says how much of that comes from the image.
"""
from __future__ import annotations

import os
from pathlib import Path

import argparse
import glob
import json
import random
from collections import Counter, defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("MEDVIGIL3D_ROOT", str(Path(__file__).resolve().parent.parent)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=3.0,
                    help="both gaps must clear the growth amount by this much, "
                         "so neither member sits on the decision boundary")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # (volume, lesion) -> [(target, gap)]
    by_lesion = defaultdict(list)
    organ_of = {}
    for f in glob.glob(f"{args.root}/cfqa_*/qa/*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("kind") != "growth_contact" or not r.get("pair_id"):
                continue
            p = r["provenance"]
            if p.get("gap_mm") is None:
                continue
            key = r["pair_id"].rsplit("_" + p["target"], 1)[0]
            by_lesion[key].append((p["target"], float(p["gap_mm"])))
            organ_of[key] = r.get("organ") or f.split("/")[-3].replace("cfqa_", "")

    rng = random.Random(args.seed)
    rows, pairs = [], 0
    for key, ts in sorted(by_lesion.items()):
        ts = sorted(set(ts), key=lambda x: x[1])
        if len(ts) < 2:
            continue
        # choose a growth amount that separates two targets with margin on both
        cands = []
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                near, far = ts[i], ts[j]
                if far[1] - near[1] < 2 * args.margin:
                    continue
                G = round((near[1] + far[1]) / 2, 1)
                if G - near[1] >= args.margin and far[1] - G >= args.margin:
                    cands.append((G, near, far))
        if not cands:
            continue
        G, near, far = rng.choice(cands)
        vid = "_".join(key.split("_")[:2])
        pid = f"{key}_tp{G:g}"
        for tgt, gap, ans in ((near[0], near[1], "yes"), (far[0], far[1], "no")):
            assert (gap <= G) == (ans == "yes")
            rows.append({
                "qid": f"{pid}_{tgt}", "kind": "target_contrast",
                "question": (f"If this lesion grew by {G:g} mm in every "
                             f"direction, would it contact the "
                             f"{tgt.replace('_',' ')}? Answer yes or no."),
                "answer": ans, "choices": ["no", "yes"], "pair_id": pid,
                "organ": organ_of[key],
                "provenance": {"target": tgt, "gap_mm": gap, "growth_mm": G,
                               "rule": "surface distance after isotropic growth",
                               "contrast": "target"}})
        pairs += 1
        if len(rows) >= args.n:
            break

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lab = Counter(r["answer"] for r in rows)
    gs = [r["provenance"]["growth_mm"] for r in rows]
    byl = defaultdict(list)
    for r in rows:
        byl[r["answer"]].append(r["provenance"]["growth_mm"])
    import statistics as st
    print(f"wrote {len(rows)} probes over {pairs} target-contrast pairs")
    print(f"  labels: {dict(lab)}")
    print(f"  growth median  yes {st.median(byl['yes']):.1f} mm   "
          f"no {st.median(byl['no']):.1f} mm")
    # the property this construction exists to have
    assert lab["yes"] == lab["no"], "pairs must be label-balanced"
    assert st.median(byl["yes"]) == st.median(byl["no"]), \
        "growth distributions must be identical across labels by construction"
    best = max(sum((g > t) == (r["answer"] == "yes")
                   for g, r in zip(gs, rows)) / len(rows)
               for t in sorted(set(gs)))
    print(f"  growth-threshold ceiling: {100*best:.1f}%  "
          f"(50.0% = the numeric shortcut is gone)")
    assert best < 0.55, "a growth threshold still separates the labels"


if __name__ == "__main__":
    main()
