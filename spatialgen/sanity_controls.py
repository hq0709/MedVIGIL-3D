"""
Model-level sanity controls: questions whose answers are known without expertise.

Why this belongs in the paper rather than in a debugging notebook
-----------------------------------------------------------------
Med3DVLM and M3D-LaMed both score exactly 50.0 % on the counterfactual probes.
Read from accuracy alone they are the same result. They are not: one is a
spatial-reasoning failure, the other is a binary channel stuck on "no" that
also answers "no" to "Is this a CT scan?". No amount of task accuracy can
separate those, because a stuck channel produces chance-level accuracy on any
balanced task whatsoever.

These controls are the separator. They carry no clinical content -- a model that
cannot answer "is this a CT scan" about a CT scan has not failed at radiology,
it has failed at responding -- and any model that misses them cannot be
interpreted on the main benchmark at all.

Scored identically to the benchmark (likelihood over option strings), so a
failure here cannot be blamed on a different scoring path.
"""
from __future__ import annotations

import os

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_multimodel import FAMILY, PROMPT, MontageModel  # noqa: E402

# (question, true answer, what a wrong answer implies)
CONTROLS = [
    ("Is this a CT scan?", "yes", "modality"),
    ("Is this a photograph of a cat?", "no", "modality"),
    ("Does this image show a brain MRI?", "no", "modality"),
    ("Are there any bones in this image?", "yes", "content"),
    ("Is the spine visible in this image?", "yes", "content"),
    ("Is the patient's head visible in this image?", "no", "content"),
    ("Is this image completely blank?", "no", "content"),
]


# The montage prompt describes the input as "axial, coronal and sagittal views".
# That is true for a model receiving a montage and false for one receiving the
# volume, and one of the control questions asks whether the image is a CT scan --
# which the montage prompt answers for the model. A native model was therefore
# judged under a description of an input it never got. This is the same question
# set with the framing its input actually has.
NATIVE_PROMPT = ("This is a CT volume. {q} Answer with exactly one of: {opts}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=os.environ.get("MSD_ROOT", ""))
    ap.add_argument("--organ", default="Task03_Liver")
    ap.add_argument("--volumes", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    fam = FAMILY.get(args.model, "qwen")
    vols = sorted(glob.glob(
        f"{args.data_root}/{args.organ}/imagesTr/*.nii.gz"))[: args.volumes]
    print(f"{len(CONTROLS)} controls x {len(vols)} volumes / family={fam}",
          flush=True)

    if fam == "m3d":
        from m3d_infer import M3D, preprocess_volume
        model = M3D(device=args.device, model_id=args.model)
    elif fam == "med3dvlm":
        from med3dvlm_infer import Med3DVLM, preprocess_volume
        model = Med3DVLM(device=args.device)
    else:
        model = MontageModel(args.model, args.device)
    print("model ready", flush=True)

    from PIL import Image

    from render import montage, orthogonal_views
    from scene_graph import load_ras

    rows = []
    with open(args.out, "w") as f:
        for vp in vols:
            if fam in ("m3d", "med3dvlm"):
                prepared = preprocess_volume(vp)
            else:
                vol, affine = load_ras(vp)
                r = orthogonal_views(vol, np.abs(np.diag(affine)[:3]))
                prepared = Image.fromarray(montage(r.available())).convert("RGB")
            for q, gold, kind in CONTROLS:
                try:
                    if fam in ("m3d", "med3dvlm"):
                        pred, sc = model.score_choices(
                            NATIVE_PROMPT.format(q=q, opts="yes, no"),
                            prepared, ["yes", "no"])
                    else:
                        pred, sc = model.score(q, ["yes", "no"], prepared)
                except Exception as e:
                    print(f"  failed {q}: {e}", file=sys.stderr)
                    continue
                row = {"volume": Path(vp).stem, "question": q, "gold": gold,
                       "prediction": pred, "kind": kind,
                       "logprobs": {k: round(v, 4) for k, v in sc.items()}}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            f.flush()

    n_ok = sum(r["prediction"] == r["gold"] for r in rows)
    yes_rate = 100.0 * sum(r["prediction"] == "yes" for r in rows) / len(rows)
    margins = [abs(r["logprobs"]["yes"] - r["logprobs"]["no"]) for r in rows]
    print(f"\n通过 {n_ok}/{len(rows)} = {100*n_ok/len(rows):.1f}%   "
          f"预测 yes 占比 {yes_rate:.1f}% (健康模型应接近 43%，即 3/7)   "
          f"平均决策间隔 {np.mean(margins):.2f}")
    for q, gold, _ in CONTROLS:
        sel = [r for r in rows if r["question"] == q]
        ok = sum(r["prediction"] == r["gold"] for r in sel)
        print(f"  {q:44} 应答 {gold:>3}  通过 {ok:3d}/{len(sel)}")
    if yes_rate in (0.0, 100.0):
        print("\n判定：二值通道退化为常量 —— 该模型在本基准上的准确率无法解释")
    elif n_ok / len(rows) < 0.6:
        print("\n判定：未通过对照 —— 主基准结果需谨慎解释")
    else:
        print("\n判定：通过对照 —— 主基准结果可解释")


if __name__ == "__main__":
    main()
