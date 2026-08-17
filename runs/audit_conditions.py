"""
What does each identification-control condition actually put in front of the
model? Measured on the images, before any GPU time is spent on them.

The four conditions are meant to isolate three things: seeing the structures
(`bestslice`), knowing which they are (`overlay`), and both plus a scale bar
(`identified`). Two of those only hold if the pixels cooperate:

  * `plain` and `overlay` slice through the geometric centre of the volume. A
    lesion of a few hundred voxels usually does not intersect a centre slice, so
    in `overlay` the legend sentence "the lesion in question is outlined in red"
    can describe an outline that was never drawn.
  * `bestslice` and `identified` choose slices to maximise joint visibility, but
    two structures separated along all three axes cannot share a plane, so even
    there the guarantee is per structure across the three panels, not per panel.

An unmeasured share of mislabelled cases is not something a paper can report
around, and the share is a property of the geometry rather than of any model --
so it is cheap, and it belongs before the 64 model runs rather than after.

Runs on CPU, one process per volume.

    python runs/audit_conditions.py --organs Task03_Liver --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spatialgen"))
sys.path.insert(0, str(REPO))

CONDITIONS = ["plain", "bestslice", "overlay", "identified"]


def one_volume(job: tuple) -> list[dict]:
    organ, vid, rows, msd_root = job
    from lesion_binding import LESION_LABEL, find_lesions
    from run_identification_control import lesion_key, render
    from run_pipeline import label_map
    from scene_graph import load_ras

    task = Path(msd_root) / organ
    segp = REPO / f"cfqa_{organ}" / "seg_cache" / f"{vid}_seg.nii.gz"
    volp = task / "imagesTr" / f"{vid}.nii.gz"
    labp = task / "labelsTr" / f"{vid}.nii.gz"
    if not (segp.exists() and volp.exists() and labp.exists()):
        return []
    name2lab = {v: k for k, v in label_map().items()}
    vol, affine = load_ras(str(volp))
    gt, _ = load_ras(str(labp))
    seg, _ = load_ras(str(segp))
    spacing = np.abs(np.diag(affine)[:3])
    lesions = dict(find_lesions(gt == LESION_LABEL[organ], affine))
    vol16 = vol.astype(np.int16)

    out = []
    seen: dict[tuple, dict] = {}
    for r in rows:
        lk = lesion_key(r["qid"])
        tname = r.get("provenance", {}).get("target")
        if lk not in lesions or tname not in name2lab:
            continue
        key = (lk, tname)
        if key not in seen:
            tmask = seg == name2lab[tname]
            if not tmask.any():
                continue
            rec = {}
            for cond in CONDITIONS:
                _, geom = render(vol16, lesions[lk], tmask, spacing, cond)
                rec[cond] = geom
            rec["lesion_voxels_total"] = int(lesions[lk].sum())
            seen[key] = rec
        out.append({"qid": r["qid"], "organ": organ, **seen[key]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--organs", nargs="*",
                    default=["Task03_Liver", "Task06_Lung", "Task07_Pancreas",
                             "Task10_Colon"])
    ap.add_argument("--msd-root", default=os.environ.get("MSD_ROOT", ""))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results_new/condition_audit.jsonl")
    args = ap.parse_args()

    from growth_matched import growth_of, matched_subset
    ref = {}
    for line in open(REPO / "common_subset" / "qa" / "all.jsonl"):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
    keep = matched_subset(ref, growth_of())

    jobs = []
    for organ in args.organs:
        by_vol = defaultdict(list)
        for f in sorted((REPO / f"cfqa_{organ}" / "qa").glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["qid"] in keep:
                    by_vol["_".join(r["qid"].split("_")[:2])].append(r)
        jobs += [(organ, vid, rows, args.msd_root)
                 for vid, rows in sorted(by_vol.items())]
    print(f"{len(jobs)} volumes, {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(one_volume, jobs), 1):
            rows += res
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} volumes, {len(rows)} probes",
                      flush=True)
    with open(REPO / args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\nwrote {args.out}: {len(rows)} probes\n")
    hdr = (f"{'organ':16}{'condition':11}{'n':>6}{'lesion on slice':>17}"
           f"{'target on slice':>17}{'both':>7}{'red drawn':>11}{'cyan drawn':>12}")
    print(hdr)
    print("-" * len(hdr))
    for organ in ["ALL"] + list(args.organs):
        sub = rows if organ == "ALL" else [r for r in rows if r["organ"] == organ]
        if not sub:
            continue
        for cond in CONDITIONS:
            n = len(sub)
            lv = sum(r[cond]["lesion_voxels_shown"] > 0 for r in sub)
            tv = sum(r[cond]["target_voxels_shown"] > 0 for r in sub)
            both = sum(r[cond]["lesion_voxels_shown"] > 0
                       and r[cond]["target_voxels_shown"] > 0 for r in sub)
            red = sum(r[cond]["lesion_outline_px"] > 0 for r in sub)
            cyan = sum(r[cond]["target_outline_px"] > 0 for r in sub)
            print(f"{organ:16}{cond:11}{n:>6}{100 * lv / n:>16.1f}%"
                  f"{100 * tv / n:>16.1f}%{100 * both / n:>6.1f}%"
                  f"{100 * red / n:>10.1f}%{100 * cyan / n:>11.1f}%")
        print()
    print("`lesion on slice` is the share of probes whose lesion has at least "
          "one voxel on the three slices shown; `red drawn` is the share whose "
          "lesion outline was actually painted, which is zero by design in the "
          "unannotated conditions.")


if __name__ == "__main__":
    main()
