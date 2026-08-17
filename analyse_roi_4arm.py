"""
The four-arm ROI comparison, which refuses to report until the arms are equal.

VGR compares `roi_only` against `roi_masked`, and those two arms are not equally
intact: `roi_only` replaces everything outside the evidence region -- roughly
99 % of the volume -- while `roi_masked` replaces only the region itself. A
negative VGR is therefore consistent with two incompatible readings, and the
`full` and `zero` arms are what separate them:

    full  ~=  roi_masked  ~=  zero     the volume carries no signal on these
                                       items at all; roi_only's deficit is
                                       image degradation and VGR is measuring
                                       intactness, not grounding

    full   >  roi_masked               masking the evidence costs accuracy, so
                                       the region does carry signal and a
                                       negative VGR needs the stronger reading

The guard exists because the first attempt at this comparison ran while the
`zero` arm was eight rows into a six-hundred-item run. Intersecting the four
arms gave n = 8, and the resulting 62.5 % / 50.0 % / 75.0 % / 50.0 % looked like
a result. Nothing in the arithmetic objects to n = 8; the script has to.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import os as _os
R = _os.environ.get("MEDVIGIL3D_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
ARMS = ["full", "roi_only", "roi_masked", "zero"]
ORGANS = ["Task06_Lung", "Task10_Colon", "Task07_Pancreas", "Task03_Liver"]



def _being_written(path: str) -> bool:
    """A file a runner still holds is unfinished, not mismatched."""
    return subprocess.run(["pgrep", "-af", "run_roi_arms.py"],
                          capture_output=True, text=True
                          ).stdout.find(os.path.basename(path)) >= 0


def arm_path(T: str, model: str, arm: str) -> str:
    """Prefer the explicitly-fill-labelled file over the un-suffixed one.

    The re-measured arms are written as ..._roi_only_local.jsonl; the older
    un-suffixed files inherited whatever the argparse default was when their
    process started and cannot be attested. Mixing the two inside one four-arm
    comparison would reintroduce exactly the provenance problem the re-run
    exists to remove, so the labelled file wins whenever it exists.
    """
    labelled = f"{R}roi_{T}_{model}_{arm}_local.jsonl"
    return labelled if os.path.exists(labelled) else f"{R}roi_{T}_{model}_{arm}.jsonl"

def load(p: str):
    return ({json.loads(l)["qid"]: json.loads(l) for l in open(p)}
            if os.path.exists(p) else None)


def boot(D, keys, n=2000, seed=0):
    byv = defaultdict(list)
    for q in keys:
        byv["_".join(q.split("_")[:2])].append(q)
    vs = sorted(byv)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [vs[rng.randrange(len(vs))] for _ in vs]
        qs = [q for v in s for q in byv[v]]
        out.append(100 * sum(D[q]["prediction"] == D[q]["gold"] for q in qs) / len(qs))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen32b"
    ready = 0
    seen_fill: dict = {}
    for T in ORGANS:
        arms = {c: load(arm_path(T, model, c)) for c in ARMS}
        present = {c: d for c, d in arms.items() if d}
        if len(present) < 4:
            print(f"{T:16} waiting — arms present: "
                  f"{', '.join(f'{c}({len(d)})' for c, d in present.items()) or 'none'}")
            continue
        sizes = {c: len(d) for c, d in present.items()}
        live = [c for c, d in present.items()
                if _being_written(arm_path(T, model, c))]
        if live and len(set(sizes.values())) != 1:
            print(f"{T:16} in progress — {sizes}  (writing: {', '.join(live)})")
            continue
        if len(set(sizes.values())) != 1:
            print(f"{T:16} INCOMPLETE — arms disagree in size: {sizes}; "
                  f"a four-way comparison over the intersection would be taken "
                  f"on {len(set.intersection(*[set(d) for d in present.values()]))} items")
            continue
        # Every arm in one row must share a fill, and every row in one table
        # must too. The re-measured arms record `fill` per row; the older
        # un-suffixed files predate that field and are air-filled. Mixing them
        # silently produced a table whose InternVL rows were air and whose
        # Qwen-32B rows were local -- a difference of up to 1.8 pp per arm,
        # which is the size of the effect the table reports.
        # Only the two masked arms consult the fill: `full` copies the volume
        # and `zero` zeroes it, so the parameter is recorded for them but has
        # no effect. Comparing all four flagged every organ as mixed.
        fills = set()
        for c in ("roi_only", "roi_masked"):
            f = next(iter(present[c].values())).get("fill")
            fills.add(f if f else "unrecorded")
        # "unrecorded" is not a fill, it is the absence of evidence about one.
        # The two masked arms of one VGR must agree, and agreeing on "we do not
        # know" is not agreement -- Task03_Liver/qwen32b paired a local-fill arm
        # against an unrecorded one and the difference was published.
        if len(fills) > 1 or fills == {"unrecorded"}:
            what = ("MIXED FILLS" if len(fills) > 1
                    else "FILL NOT RECORDED in either arm")
            print(f"{T:16} {what}: {sorted(fills)} — not attestable")
            continue
        seen_fill.setdefault(model, set()).add(next(iter(fills)))

        keys = sorted(set.intersection(*[set(d) for d in present.values()]))
        assert len(keys) == list(sizes.values())[0], "arms cover different items"
        ready += 1

        def acc(c):
            return 100 * sum(present[c][q]["prediction"] == present[c][q]["gold"]
                             for q in keys) / len(keys)

        print(f"\n{T}  ({model}, n = {len(keys)})")
        for c in ARMS:
            lo, hi = boot(present[c], keys)
            print(f"  {c:11} {acc(c):5.1f} %  [{lo:.1f}, {hi:.1f}]")
        print(f"  VGR (only − masked)           {acc('roi_only') - acc('roi_masked'):+6.1f} pp")
        print(f"  cost of masking (full − masked){acc('full') - acc('roi_masked'):+6.1f} pp")
        print(f"  cost of the rest (full − only) {acc('full') - acc('roi_only'):+6.1f} pp")
        print(f"  signal at all (full − zero)    {acc('full') - acc('zero'):+6.1f} pp")

    for m, fs in seen_fill.items():
        if len(fs) > 1:
            print(f"\n{m}: organs do not share a fill ({sorted(fs)}); the rows "
                  f"are not comparable to each other")
            sys.exit(3)
        elif fs:
            print(f"\n{m}: all reported organs use fill={next(iter(fs))}")
    if not ready:
        print("\nno organ has all four arms at equal length — nothing to report")
        sys.exit(2)


if __name__ == "__main__":
    main()
