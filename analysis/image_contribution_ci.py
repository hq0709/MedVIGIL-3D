"""
Paired confidence intervals on the volumetric image contribution.

The round-2 review objected to "none was measurably helped by the volume",
correctly: Table 4 gave point estimates for the sighted-minus-blind difference
with no uncertainty attached, so the sentence asserted more than the table
showed. This computes the missing intervals.

The contrast is paired by construction -- the same probe is scored twice, once
with the montage or volume and once without -- so the resampling has to preserve
that pairing, and it has to resample at the level the probes are clustered by.
Probes are nested in volumes, so the unit of resampling is the volume: draw
volumes with replacement, take every probe belonging to a drawn volume, and
recompute the paired difference within the draw.

Reuses the growth-matched subset from growth_matched.py rather than
re-deriving it, so the items are exactly those the accuracy table reports on.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = os.path.join(BASE, "from3d")
OUT = os.path.join(BASE, "out")
sys.path.insert(0, D3)

from growth_matched import ORDER, LABEL, growth_of, matched_subset  # noqa: E402

B = 10000
SEED = 0
NATIVE = {"m3d", "m3dllama", "med3dvlm"}


def volume_of(qid: str) -> str:
    return "_".join(qid.split("_")[:2])


def load(path: str):
    if not os.path.exists(path):
        return None
    return {json.loads(l)["qid"]: json.loads(l) for l in open(path) if l.strip()}


def main() -> None:
    ref = {}
    for line in open(os.path.join(D3, "common_subset", "qa", "all.jsonl")):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
    keep = matched_subset(ref, growth_of())

    rng = np.random.default_rng(SEED)
    rows = []
    for name in ORDER:
        S = load(os.path.join(D3, f"mm_{name}_sighted.jsonl"))
        Bl = load(os.path.join(D3, f"mm_{name}_blind.jsonl"))
        if not S or not Bl or len(S) != len(ref) or len(Bl) != len(ref):
            continue
        qs = sorted(q for q in keep if q in S and q in Bl)

        by_vol: dict[str, list[int]] = defaultdict(list)
        for q in qs:
            d = int(S[q]["prediction"] == S[q]["gold"]) - \
                int(Bl[q]["prediction"] == Bl[q]["gold"])
            by_vol[volume_of(q)].append(d)

        vols = sorted(by_vol)
        arrs = [np.array(by_vol[v], dtype=float) for v in vols]
        obs = 100.0 * np.concatenate(arrs).mean()

        idx = rng.integers(0, len(vols), size=(B, len(vols)))
        boot = np.empty(B)
        for i in range(B):
            boot[i] = 100.0 * np.concatenate([arrs[j] for j in idx[i]]).mean()

        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "model": LABEL[name],
            "input": "native" if name in NATIVE else "montage",
            "n_probes": len(qs), "n_volumes": len(vols),
            "image_contribution": round(obs, 2),
            "ci_lo": round(float(lo), 2), "ci_hi": round(float(hi), 2),
            "excludes_zero": "yes" if (lo > 0 or hi < 0) else "no",
        })

    rows.sort(key=lambda r: r["image_contribution"])
    print("Image contribution on the growth-matched subset "
          "(sighted minus blind, identical probes)")
    print(f"Volume-level paired bootstrap, B={B:,}, seed={SEED}\n")
    print(f"{'model':24}{'input':>9}{'contribution':>14}{'95% CI':>18}{'excl. 0':>10}")
    print("-" * 76)
    for r in rows:
        ci = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
        print(f"{r['model']:24}{r['input']:>9}{r['image_contribution']:+14.1f}"
              f"{ci:>18}{r['excludes_zero']:>10}")

    n_excl = sum(r["excludes_zero"] == "yes" for r in rows)
    print(f"\n{len(rows)} models; {r['n_probes']} probes over {r['n_volumes']} volumes")
    print(f"contribution range {min(r['image_contribution'] for r in rows):+.1f} to "
          f"{max(r['image_contribution'] for r in rows):+.1f} pp")
    print(f"intervals excluding zero: {n_excl}/{len(rows)}")
    if n_excl:
        for r in rows:
            if r["excludes_zero"] == "yes":
                print(f"  {r['model']}: {r['image_contribution']:+.1f} "
                      f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]")

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "image_contribution_ci.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
