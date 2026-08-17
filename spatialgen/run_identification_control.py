"""
Identification control: does the published null survive telling the model which
lesion and which target the question is about?

Why this exists
---------------
Every volumetric result in the audit was produced by feeding the model
`montage(orthogonal_views(volume))`. Read `render.orthogonal_views`: with
`index=None` it takes three slices through the **geometric centre of the
volume**, and it draws no annotation. The question, meanwhile, is

    "If this lesion grew by 5.8 mm in every direction, would it contact the heart?"

and the qid is `liver_0_lesion5_heart_g5.8` -- there are at least five lesions in
that liver. Nothing in the image says which one is "this lesion", nothing marks
the target, the lesion need not intersect any of the three centre slices at all,
and no scale bar is drawn although the question is metric.

So "no model's CI excluded chance" currently admits a mundane reading that has
nothing to do with geometric reasoning: **the probe may be underspecified given
the input the model received.** A reviewer will say this in one sentence.

The reader study does not settle it either. `export_reader_study.render_case`
gives the radiologist lesion-in-red, target-in-cyan, slices chosen to maximise
joint visibility, and a 10 mm scale bar per panel. That is strictly more
information than any model ever got, so a reader-versus-model comparison on those
two inputs is not a comparison of reasoning.

This script closes the gap by running the same probes under a small factorial of
input conditions, so that any change can be attributed:

    plain       centre slices, no annotation, no scale bar
                -- reproduces the published condition, and is the control
    bestslice   slices chosen to show lesion and target, still no annotation
                -- does the model just need to SEE the structures?
    overlay     centre slices, lesion and target outlined
                -- does the model just need to know WHICH structures?
    identified  best slices + outlines + 10 mm scale bar
                -- exactly what the radiologist sees

Interpreting the outcome
------------------------
If `identified` stays at chance, the null is about geometric reasoning and it is
defensible: the model was given everything the reader was given and still could
not answer. If `identified` rises above chance, the published null was
substantially an artefact of input specification, and the claim must be weakened.
Either result is worth having, and it is much better to find the second one here
than in review.

Usage
-----
    python run_identification_control.py \\
        --qa      ../cfqa_Task03_Liver/qa/all.jsonl \\
        --task-dir /path/to/MSD/Task03_Liver \\
        --seg-cache ../cfqa_Task03_Liver/seg_cache \\
        --model   qwen32b \\
        --condition identified \\
        --subset  matched \\
        --out     ../id_Task03_Liver_qwen32b_identified.jsonl

Output format matches the existing `mm_*`/`roi_*` runs, so the analysis scripts
in this repository read it without modification.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_reader_study import _best_slice, outline           # noqa: E402
from lesion_binding import LESION_LABEL, find_lesions          # noqa: E402
from render import window                                      # noqa: E402
from scene_graph import load_ras                               # noqa: E402

CONDITIONS = ["plain", "bestslice", "overlay", "identified"]
LESION_RGB = [255, 60, 60]
TARGET_RGB = [60, 200, 255]


def lesion_key(qid: str) -> str | None:
    for part in qid.split("_"):
        if part.startswith("lesion"):
            return part
    return None


def render(vol: np.ndarray, lesion: np.ndarray, target: np.ndarray,
           spacing: np.ndarray, condition: str,
           preset: str = "soft_tissue") -> np.ndarray:
    """Three orthogonal panels under one of the four input conditions.

    Panel geometry and colours match export_reader_study.render_case exactly, so
    the `identified` condition is byte-comparable with what the reader saw.
    """
    axes = [2, 1, 0]
    if condition in ("bestslice", "identified"):
        idx = [_best_slice(lesion, a, target) for a in axes]
        # deterministic repair, as in render_case: a structure absent from all
        # three chosen slices gets a panel re-pointed at it
        def seen(m):
            out = False
            for a, k in zip(axes, idx):
                sl = [slice(None)] * 3
                sl[a] = k
                out = out or bool(m[tuple(sl)].sum())
            return out
        if not seen(lesion):
            idx[0] = _best_slice(lesion, axes[0])
        if not seen(target):
            idx[1] = _best_slice(target, axes[1])
    else:
        idx = [vol.shape[a] // 2 for a in axes]

    annotate = condition in ("overlay", "identified")
    scalebar = condition == "identified"
    grey = window(vol, preset)

    panels = []
    for a, k in zip(axes, idx):
        sl = [slice(None)] * 3
        sl[a] = int(np.clip(k, 0, vol.shape[a] - 1))
        g = grey[tuple(sl)]
        le = outline(lesion[tuple(sl)])
        tg = outline(target[tuple(sl)])
        g, le, tg = (x.T[::-1] for x in (g, le, tg))
        rgb = np.dstack([g] * 3)
        if annotate:
            rgb[le] = LESION_RGB
            rgb[tg] = TARGET_RGB
        if scalebar:
            ip = [sp for i, sp in enumerate(spacing) if i != a][0]
            n = max(4, int(round(10.0 / float(ip))))
            h, w = rgb.shape[:2]
            rgb[h - 8:h - 5, 6:6 + min(n, w - 12)] = [255, 255, 255]
        panels.append(rgb)

    h = max(p.shape[0] for p in panels)
    pad = 8
    out = np.zeros((h, sum(p.shape[1] for p in panels) + pad * (len(panels) - 1), 3),
                   np.uint8)
    x = 0
    for p in panels:
        out[: p.shape[0], x: x + p.shape[1]] = p
        x += p.shape[1] + pad
    return out


def legend_for(condition: str) -> str:
    """Annotation legend only.

    MontageModel.score() already wraps whatever it is handed in
    run_multimodel.PROMPT ("This is a CT scan shown as axial, coronal and
    sagittal views. {q} Answer with exactly one of: {opts}."), so this must NOT
    repeat the view description or the answer instruction -- doing so double-wraps
    the prompt and changes the input in a way unrelated to the condition.

    Describing an annotation the model did not receive would be its own confound,
    so each sentence is added only when the corresponding pixels are drawn.
    """
    parts = []
    if condition in ("overlay", "identified"):
        parts.append("The lesion in question is outlined in red and the target "
                     "structure is outlined in cyan.")
    if condition == "identified":
        parts.append("The white bar in each panel is 10 mm in that panel's plane.")
    return " ".join(parts)


def matched_subset_keys(qa_path: Path) -> set[str] | None:
    """Reuse growth_matched.py's subset so results are directly comparable with
    the published table rather than a differently-drawn subset."""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    try:
        from growth_matched import growth_of, matched_subset
    except Exception as exc:                    # pragma: no cover
        print(f"could not import growth_matched ({exc}); running all items",
              file=sys.stderr)
        return None
    ref = {}
    common = root / "common_subset" / "qa" / "all.jsonl"
    if not common.exists():
        print(f"missing {common}; running all items", file=sys.stderr)
        return None
    for line in open(common):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
    return matched_subset(ref, growth_of())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--qa", required=True, help="probe jsonl (or a directory of them)")
    ap.add_argument("--task-dir", required=True, help="MSD task dir with imagesTr/labelsTr")
    ap.add_argument("--seg-cache", required=True, help="anatomy segmentation cache")
    ap.add_argument("--model", required=True, help="tag from run_multimodel.MODEL_ID")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--subset", choices=["matched", "all"], default="matched")
    ap.add_argument("--device", default="cuda:0",
                    help='concrete device ("cuda:0") or "auto" to shard a model too large for one card across every visible GPU')
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (smoke tests)")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    files = sorted(qa_path.glob("*.jsonl")) if qa_path.is_dir() else [qa_path]
    items = [json.loads(l) for f in files for l in open(f) if l.strip()]

    if args.subset == "matched":
        keep = matched_subset_keys(qa_path)
        if keep is not None:
            before = len(items)
            items = [r for r in items if r["qid"] in keep]
            print(f"growth-matched subset: {len(items)} of {before} probes", flush=True)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no probes selected")

    task = Path(args.task_dir)
    organ = task.name
    lesion_label = LESION_LABEL[organ]

    from run_multimodel import MODEL_ID, MontageModel
    if args.model not in MODEL_ID:
        raise SystemExit(f"unknown model tag {args.model!r}; known: "
                         + ", ".join(sorted(MODEL_ID)))
    from PIL import Image
    model = MontageModel(MODEL_ID[args.model], args.device)
    print("model ready", flush=True)

    by_vol: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        by_vol["_".join(r["qid"].split("_")[:2])].append(r)

    from run_pipeline import label_map
    name2lab = {v: k for k, v in label_map().items()}
    legend = legend_for(args.condition)
    written = skipped = 0
    with open(args.out, "w") as fout:
        for vid, group in sorted(by_vol.items()):
            volp = task / "imagesTr" / f"{vid}.nii.gz"
            labp = task / "labelsTr" / f"{vid}.nii.gz"
            segp = Path(args.seg_cache) / f"{vid}_seg.nii.gz"
            if not (volp.exists() and labp.exists() and segp.exists()):
                skipped += len(group)
                continue

            vol, affine = load_ras(str(volp))
            gt, _ = load_ras(str(labp))
            seg, _ = load_ras(str(segp))
            spacing = np.abs(np.diag(affine)[:3])
            lesions = dict(find_lesions(gt == lesion_label, affine))

            # one render per (lesion, target); the matched pair shares it
            cache: dict[tuple[str, str], object] = {}
            for r in group:
                lk = lesion_key(r["qid"])
                tname = r.get("provenance", {}).get("target")
                if lk is None or lk not in lesions or tname not in name2lab:
                    skipped += 1
                    continue
                ck = (lk, tname)
                if ck not in cache:
                    tmask = seg == name2lab[tname]
                    if not tmask.any():
                        skipped += 1
                        continue
                    img = render(vol.astype(np.int16), lesions[lk],
                                 tmask, spacing, args.condition)
                    cache[ck] = Image.fromarray(img).convert("RGB")
                choices = r.get("choices") or ["no", "yes"]
                asked = (f"{legend} {r['question']}" if legend else r["question"])
                pred, logprobs = model.score(asked, choices, cache[ck])
                fout.write(json.dumps({
                    "qid": r["qid"], "organ": organ, "condition": args.condition,
                    "prediction": pred, "gold": r["answer"],
                    "pair_id": r.get("pair_id"), "logprobs": logprobs,
                }) + "\n")
                written += 1
            if written and written % 200 < len(group):
                print(f"  {written} written", flush=True)

    print(f"wrote {args.out}: {written} scored, {skipped} skipped")
    if skipped:
        print("  skips are volumes or targets absent from the caches; "
              "they are excluded from both arms so the comparison stays paired")


if __name__ == "__main__":
    main()
