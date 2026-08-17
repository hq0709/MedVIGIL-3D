"""
Emit the job list for the identification control (E1) and the ceiling arms (E2).

Scope comes straight from EXPERIMENTS_TO_RUN.md: four conditions x four organs x
four models on the growth-matched subset (64 runs), plus the two text ceiling
arms per model on the common subset. `qwen7b` is in the model list on purpose --
it answered "no" to all 2,262 probes, so it is the cleanest test of whether
identification unsticks a degenerate responder.

VRAM figures are bf16 weights from the table in EXPERIMENTS_TO_RUN.md plus
measured runtime overhead, and they are what the queue schedules on, so they are
listed here rather than guessed per launch.

    python runs/make_jobs.py --out runs/jobs_e1.jsonl              # E1, full
    python runs/make_jobs.py --limit 20 --organs Task03_Liver \
        --models qwen32b --out runs/jobs_smoke.jsonl                # wave 0
"""
from __future__ import annotations

import argparse
import json
import os

ORGANS = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task10_Colon"]
MODELS = ["qwen32b", "internvl", "qwen3vl", "qwen7b"]
IMAGE_CONDITIONS = ["plain", "bestslice", "overlay", "identified"]
TEXT_CONDITIONS = ["text-oracle", "text-oracle-blind"]

# bf16 weights plus MEASURED peak reservation on this corpus, which is what the
# queue schedules on. The gap is much larger than a rounding allowance and it is
# not about weights: montages vary in size from volume to volume, so the caching
# allocator sees a new block shape constantly and fragments. Measured, a 7B model
# with 14 GB of weights sat on 41.8 GB. Two things bring that down -- scoring
# only the option's own logit rows instead of casting the whole
# sequence x 152k-vocabulary tensor to float32 (see MontageModel.score), and
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- and the numbers below are
# the post-fix reservations with headroom for the largest montages.
VRAM_GB = {"qwen72b": 150, "qwen32b": 70, "aria": 56, "internvl14": 40,
           "pixtral": 34, "internvl": 20, "qwen3vl": 20, "qwen7b": 20,
           "qwen3b": 14, "smolvlm": 10}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--msd-root", default=os.environ.get("MSD_ROOT", ""))
    ap.add_argument("--organs", nargs="*", default=ORGANS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--conditions", nargs="*", default=IMAGE_CONDITIONS)
    ap.add_argument("--text-arms", action="store_true",
                    help="also emit the E2 text ceiling arms (common subset)")
    ap.add_argument("--text-conditions", nargs="*", default=TEXT_CONDITIONS,
                    help="override which ceiling arms --text-arms emits")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--results", default="results_new")
    ap.add_argument("--tag", default="", help="suffix on output filenames, "
                                             "e.g. _smoke")
    args = ap.parse_args()
    if not args.msd_root and args.conditions:
        raise SystemExit("--msd-root (or $MSD_ROOT) is required for the image "
                         "conditions")

    jobs = []
    for model in args.models:
        for organ in args.organs:
            for cond in args.conditions:
                label = f"id_{organ}_{model}_{cond}{args.tag}"
                jobs.append({
                    "label": label,
                    "out": f"{args.results}/{label}.jsonl",
                    "vram_gb": VRAM_GB[model],
                    "cmd": ["python", "spatialgen/run_identification_control.py",
                            "--qa", f"cfqa_{organ}/qa",
                            "--task-dir", f"{args.msd_root}/{organ}",
                            "--seg-cache", f"cfqa_{organ}/seg_cache",
                            "--model", model, "--condition", cond,
                            "--subset", "matched", "--device", "cuda:0",
                            "--out", "{OUT}"]
                    + (["--limit", str(args.limit)] if args.limit else []),
                })
        if args.text_arms:
            for cond in args.text_conditions:
                label = f"id_common_{model}_{cond}{args.tag}"
                jobs.append({
                    "label": label,
                    "out": f"{args.results}/{label}.jsonl",
                    "vram_gb": VRAM_GB[model],
                    "cmd": ["python", "spatialgen/run_identification_control.py",
                            "--qa", "common_subset/qa/all.jsonl",
                            "--model", model, "--condition", cond,
                            "--subset", "matched", "--device", "cuda:0",
                            "--out", "{OUT}"]
                    + (["--limit", str(args.limit)] if args.limit else []),
                })

    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {args.out}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
