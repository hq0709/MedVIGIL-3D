"""
Lesion-binding driver for Medical Segmentation Decathlon tasks.

Pairs an MSD lesion mask (imagesTr/labelsTr) with a TotalSegmentator anatomy
segmentation of the same volume, and emits finding->anatomy QA whose ground
truth is computed, not read out of a radiology report.

  python run_lesion.py --task ~/medai-research/data/Task06_Lung \
                       --outdir ~/medai-research/out_lung --fast --device gpu:1

Why this is the flagship rather than CT-SpatialVQA: auditing the real
CT-SpatialVQA release showed only 6.7% of its QA is directly geometry-verifiable
and 19.4% is anatomy-frame dependent, because its questions are about findings
whose ground truth comes from GPT-4o reading a report. Here the finding has a
mask, so "which lobe is the tumour in" has a checkable answer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scene_graph import build_structures, load_ras  # noqa: E402
from qa_gen import estimate_midline_x  # noqa: E402
from lesion_binding import (  # noqa: E402
    LESION_LABEL, bind_lesion, find_lesions, lesion_qa,
)
from run_pipeline import label_map, segment  # noqa: E402


def process(
    image: Path, label: Path, outdir: Path, lesion_label: int,
    fast: bool, device: str,
) -> dict:
    t0 = time.time()
    vid = image.name.split(".")[0]

    seg_path = segment(image, outdir / "seg_cache", fast, device)
    anat, affine = load_ras(str(seg_path))
    gt, gt_affine = load_ras(str(label))

    if gt.shape != anat.shape:
        raise ValueError(f"shape mismatch: label {gt.shape} vs anatomy {anat.shape}")
    # both went through as_closest_canonical, so affines must agree; a mismatch
    # here means the lesion would be bound to the wrong anatomy
    if not np.allclose(gt_affine, affine, atol=1e-3):
        raise ValueError("affine mismatch between label and anatomy segmentation")

    structures = build_structures(anat, affine, label_map())
    if not structures:
        raise ValueError("no anatomical structures segmented")
    midline = estimate_midline_x(structures)

    lesions, qas = [], []
    for lid, mask in find_lesions(gt == lesion_label, affine):
        les = bind_lesion(lid, mask, anat, affine, label_map(),
                          structures, midline)
        lesions.append(les)
        qas.extend(lesion_qa(les, vid))

    (outdir / "lesions").mkdir(parents=True, exist_ok=True)
    (outdir / "qa").mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    with open(outdir / "lesions" / f"{vid}.json", "w") as f:
        json.dump([asdict(l) for l in lesions], f, indent=1)
    with open(outdir / "qa" / f"{vid}.jsonl", "w") as f:
        for q in qas:
            f.write(json.dumps(q) + "\n")

    n_amb = sum(1 for l in lesions if l.ambiguous)
    return {"volume_id": vid, "n_lesions": len(lesions), "n_ambiguous": n_amb,
            "n_qa": len(qas), "n_structures": len(structures),
            "containers": [l.container for l in lesions if not l.ambiguous],
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="MSD task dir (has imagesTr/labelsTr)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="gpu:1")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    task = Path(args.task)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lesion_label = LESION_LABEL.get(task.name)
    if lesion_label is None:
        sys.exit(f"unknown MSD task {task.name}; add it to LESION_LABEL")

    images = sorted(p for p in (task / "imagesTr").glob("*.nii*")
                    if not p.name.startswith("._"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"no images under {task/'imagesTr'}")

    print(f"{task.name}: {len(images)} volumes, lesion label {lesion_label}",
          flush=True)

    stats, failures = [], []
    for i, img in enumerate(images, 1):
        lab = task / "labelsTr" / img.name
        if not lab.exists():
            failures.append({"volume": img.name, "error": "no label file"})
            continue
        try:
            s = process(img, lab, outdir, lesion_label, args.fast, args.device)
            stats.append(s)
            print(f"[{i}/{len(images)}] {s['volume_id']}: {s['n_lesions']} lesions "
                  f"({s['n_ambiguous']} ambiguous) -> {s['n_qa']} QA, "
                  f"containers={s['containers']} ({s['seconds']}s)", flush=True)
        except Exception as e:
            failures.append({"volume": img.name, "error": repr(e)})
            print(f"[{i}/{len(images)}] FAILED {img.name}: {e}", flush=True)

    with open(outdir / "summary.json", "w") as f:
        json.dump({"stats": stats, "failures": failures}, f, indent=1)

    if stats:
        from collections import Counter
        n_les = sum(s["n_lesions"] for s in stats)
        n_amb = sum(s["n_ambiguous"] for s in stats)
        cont = Counter(c for s in stats for c in s["containers"])
        print(f"\n{len(stats)} ok / {len(failures)} failed | "
              f"{n_les} lesions, {n_amb} ambiguous ({n_amb/max(n_les,1):.1%}), "
              f"{sum(s['n_qa'] for s in stats)} QA")
        print("container distribution:", cont.most_common(10))
        # the ambiguity rate is a result worth reporting: it is the fraction of
        # findings for which a single anatomical answer is not well defined,
        # i.e. where a report-derived benchmark would be forcing a label
        if failures:
            print(f"failure rate: {len(failures)}/{len(images)} "
                  f"= {len(failures)/len(images):.1%}")


if __name__ == "__main__":
    main()
