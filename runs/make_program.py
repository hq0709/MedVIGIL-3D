"""
Emit the remaining experiment program as queue job lists, in the order the
findings make useful rather than the order the tiers were written in.

Why this order
--------------
The result that reorganises the rest is that the models' decision variable
tracks the growth amount in the question text and carries no information from
the image: holding the sentence fixed and swapping the volume gives AUROC
0.462-0.529 across all thirteen models, while every model scores 100% on the
bare numeric comparison and ~50% once the same two numbers are wrapped in the
clinical sentence. So:

  wave 1  E3 sub-tasks + the text chain-of-thought arm. Locates the failure
          among perceive / identify / measure / compose, and asks whether
          letting the model reason repairs the wording collapse. Cheap, and it
          is the arm that decides how the mechanism sentence is written.
  wave 2  E9 input richness. The one remaining mundane explanation for an image
          channel that does nothing is that three planes do not contain the
          geometry. This tests it directly, including a magnified arm that
          holds the input budget fixed and only changes the framing.
  wave 3  E5 inference-time compute on the images, E12 leakage, E6 native
          framing on one stack.
  wave 4  E10 Aria (completes the 14th model) and E8, the four-arm grounding
          re-run with the mask fill recorded per item -- which is all that
          analyse_roi_4arm.py needs to stop refusing.

E4's 72B image arms are NOT here: they need both cards sharded, and GPU 6 is
shared with another user's job. They go in runs/jobs_e4_72b.sh, to be run when
that card is free.

    python runs/make_program.py
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSD = os.environ.get("MSD_ROOT", "/raid/home/CAMCA/hj880/msd")
ORGANS = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task10_Colon"]
R = "results_new"

# measured reservations; axial25 is 6.7 MPx and roughly 8,500 vision tokens
VRAM = {"qwen32b": 70, "internvl": 20, "qwen3vl": 20, "qwen7b": 20, "aria": 56}
VRAM_RICH = {("internvl", "axial25"): 44, ("internvl", "slices9"): 30,
             ("qwen32b", "axial25"): 73, ("qwen32b", "slices9"): 74}


def job(label, cmd, vram):
    return {"label": label, "out": f"{R}/{label}.jsonl", "vram_gb": vram,
            "cmd": cmd + ["--out", "{OUT}"]}


def write(name, jobs):
    p = os.path.join(REPO, "runs", name)
    with open(p, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{name:28} {len(jobs):>3} jobs")


def main() -> None:
    # ---- wave 2: input richness -------------------------------------------
    jobs = []
    for model in ["internvl", "qwen32b"]:
        for arm in ["slices3", "slices9", "axial25", "zoom"]:
            for organ in ORGANS:
                label = f"id_{organ}_{model}_richness-{arm}"
                jobs.append(job(label, [
                    "python", "spatialgen/run_input_richness.py",
                    "--qa", f"cfqa_{organ}/qa", "--task-dir", f"{MSD}/{organ}",
                    "--seg-cache", f"cfqa_{organ}/seg_cache", "--model", model,
                    "--arm", arm, "--subset", "matched", "--sample", "80",
                    "--device", "cuda:0"],
                    VRAM_RICH.get((model, arm), VRAM[model])))
    write("jobs_e9.jsonl", jobs)

    # ---- wave 3a: leakage --------------------------------------------------
    jobs = []
    for model in ["qwen32b", "internvl", "qwen3vl", "qwen7b"]:
        for probe in ["organ", "dataset", "caseid"]:
            label = f"leak_{model}_{probe}"
            jobs.append(job(label, [
                "python", "spatialgen/run_leakage.py", "--model", model,
                "--probe", probe, "--volumes", "80", "--device", "cuda:0"],
                VRAM[model]))
    write("jobs_e12.jsonl", jobs)

    # ---- wave 3b: native volumetric models, one stack, both framings -------
    jobs = []
    for hf, tag in [("GoodBaiBai88/M3D-LaMed-Phi-3-4B", "m3dphi3"),
                    ("GoodBaiBai88/M3D-LaMed-Llama-2-7B", "m3dllama2"),
                    ("MagicXin/Med3DVLM-Qwen-2.5-7B", "med3dvlm")]:
        for framing in ["native", "montage"]:
            label = f"sanity_{tag}_{framing}"
            jobs.append(job(label, [
                "python", "spatialgen/sanity_controls.py", "--model", hf,
                "--framing", framing, "--data-root", MSD,
                "--organ", "Task03_Liver", "--volumes", "20",
                "--device", "cuda:0"], 24))
    write("jobs_e6.jsonl", jobs)

    # ---- wave 4a: Aria, the fourteenth model -------------------------------
    jobs = []
    for arm in ["sighted", "blind"]:
        label = f"mm_aria_{arm}"
        cmd = ["python", "spatialgen/run_multimodel.py", "--model",
               "rhymes-ai/Aria", "--qa", "common_subset/qa/all.jsonl",
               "--data-root", MSD, "--device", "cuda:0"]
        if arm == "blind":
            cmd.append("--blind")
        jobs.append({"label": label, "out": f"{label}.jsonl",
                     "vram_gb": VRAM["aria"], "cmd": cmd + ["--out", "{OUT}"]})
    write("jobs_e10.jsonl", jobs)

    # ---- wave 4b: four-arm grounding, fill recorded, equal n ---------------
    # analyse_roi_4arm.py refuses on the committed files because they mix `air`,
    # `local` and unrecorded fills. The runner already writes `fill` per item, so
    # one consistent campaign is all that is needed. `local` over `air`: the
    # air fill is the volume's 1st percentile, which on CT is background air and
    # carves a cavity a model can see without reading the anatomy.
    jobs = []
    for model in ["qwen32b", "internvl", "qwen3vl", "qwen7b"]:
        for organ in ORGANS:
            for arm in ["full", "roi_only", "roi_masked", "zero"]:
                label = f"roi4_{organ}_{model}_{arm}"
                jobs.append(job(label, [
                    "python", "spatialgen/run_roi_arms.py",
                    "--qa", f"cfqa_{organ}/qa", "--task-dir", f"{MSD}/{organ}",
                    "--seg-cache", f"cfqa_{organ}/seg_cache", "--model", model,
                    "--condition", arm, "--fill", "local", "--sample", "200",
                    "--device", "cuda:0"], VRAM[model]))
    write("jobs_e8.jsonl", jobs)

    # ---- E4: needs both cards, so it is a script and not a queue job -------
    p = os.path.join(REPO, "runs", "jobs_e4_72b.sh")
    with open(p, "w") as f:
        f.write("#!/bin/bash\n"
                "# Qwen2.5-VL-72B: 146.8 GB of bf16 weights, so it needs both\n"
                "# cards sharded and cannot share either with anything else.\n"
                "# Run only when GPU 6 has no other tenant.\n"
                "set -u\n"
                "source /raid/home/CAMCA/hj880/medvigil_env.sh\n"
                "cd $MEDVIGIL3D_ROOT\n"
                "export CUDA_VISIBLE_DEVICES=6,7\n"
                "for O in " + " ".join(ORGANS) + "; do\n"
                "  for C in plain identified; do\n"
                f"    OUT={R}/id_${{O}}_qwen72b_${{C}}.jsonl\n"
                "    [ -s \"$OUT\" ] && continue\n"
                "    echo \"[$(date +%T)] qwen72b $O $C\"\n"
                "    python spatialgen/run_identification_control.py --qa cfqa_$O/qa \\\n"
                "      --task-dir $MSD_ROOT/$O --seg-cache cfqa_$O/seg_cache \\\n"
                "      --model qwen72b --condition $C --subset matched \\\n"
                "      --device auto --out $OUT.part 2>&1 | tail -2\n"
                "    [ -s \"$OUT.part\" ] && mv \"$OUT.part\" \"$OUT\"\n"
                "  done\n"
                "done\n"
                "echo \"[$(date +%T)] 72B IMAGE ARMS DONE\"\n")
    os.chmod(p, 0o755)
    print(f"{'jobs_e4_72b.sh':28}   8 runs (needs both cards; GPU 6 is shared)")


if __name__ == "__main__":
    main()
