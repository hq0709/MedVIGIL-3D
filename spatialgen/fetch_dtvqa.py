"""
Fetch a stratified slice of the DeepTumorVQA benchmark and cache it in M3D form.

Why this benchmark
------------------
CT-SpatialVQA's volumes come from CT-RATE, which is a gated HF dataset -- 401
without an approved token, so the sighted/blind test cannot be run on it here.
DeepTumorVQA 2.0 is open, ships its 991 benchmark CT volumes alongside the QA,
and is the benchmark that reported tools helping Measurement (+35.5pp) while
leaving Visual Reasoning flat (-0.4pp). Testing on it therefore attacks a
published result directly rather than one of our own construction.

Storage discipline
------------------
Full volumes are ~100-300 MB each and this box sits at 95% full on a shared
disk. Each volume is downloaded, immediately resampled to M3D's 1x32x256x256
input (8 MB as float16), and the source deleted before the next fetch. Peak extra
usage is one volume, not the whole sample.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

REPO = "tumor-vqa/DeepTumorVQA_2.0"
QA_FILE = "benchmark/test_qa.jsonl"


def load_qa() -> list[dict]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO, QA_FILE, repo_type="dataset")
    return [json.loads(l) for l in open(p) if l.strip()]


def parse_options(mc_question: str) -> tuple[str, dict[str, str]]:
    """Split 'stem A: x B: y C: z' into the stem and {letter: text}.

    Splitting on ' <LETTER>: ' rather than on the letter alone matters -- option
    text routinely contains capitals ("A: left kidney B: the same"), and a naive
    split mangles them.
    """
    import re

    parts = re.split(r"\s([A-Z]):\s", mc_question)
    stem = parts[0].strip()
    opts = {}
    for i in range(1, len(parts) - 1, 2):
        opts[parts[i]] = parts[i + 1].strip()
    return stem, opts


def stratified_sample(rows: list[dict], n_volumes: int, per_volume: int,
                      seed: int = 0) -> list[dict]:
    """Sample QA spread across volumes AND question types.

    Sampling by QA row alone would concentrate on whichever volumes happen to
    carry many questions; the effective sample size is the number of PATIENTS,
    not the number of items.
    """
    rng = random.Random(seed)
    by_vol = defaultdict(list)
    for r in rows:
        by_vol[r["image_id"]].append(r)

    vols = sorted(by_vol)
    rng.shuffle(vols)
    vols = vols[:n_volumes]

    picked = []
    for v in vols:
        items = by_vol[v]
        by_type = defaultdict(list)
        for r in items:
            by_type[r.get("question_subtype") or r["question_type"]].append(r)
        per_type = max(1, per_volume // max(len(by_type), 1))
        for t in sorted(by_type):
            chosen = by_type[t]
            rng.shuffle(chosen)
            picked.extend(chosen[:per_type])
    return picked


def fetch_and_cache(image_id: str, cache_dir: Path, keep_source: bool = False
                    ) -> Path | None:
    """Download one volume, store it as an M3D-ready array, delete the source."""
    from huggingface_hub import hf_hub_download

    from m3d_infer import preprocess_volume

    out = cache_dir / f"{image_id}.npy"
    if out.exists():
        return out

    try:
        src = hf_hub_download(REPO, f"benchmark/ct/{image_id}/ct.nii.gz",
                              repo_type="dataset")
    except Exception as e:
        print(f"  {image_id}: download failed ({type(e).__name__})", flush=True)
        return None

    try:
        arr = preprocess_volume(src).astype(np.float16)
        np.save(out, arr)
    except Exception as e:
        print(f"  {image_id}: preprocess failed ({e})", flush=True)
        return None
    finally:
        if not keep_source:
            # remove both the symlink target in the HF blob store and the link
            try:
                real = os.path.realpath(src)
                os.remove(real)
                if os.path.islink(src) or os.path.exists(src):
                    os.remove(src)
            except OSError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--volumes", type=int, default=60)
    ap.add_argument("--per-volume", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-source", action="store_true")
    ap.add_argument("--subtypes", default=None,
                    help="comma-separated question_subtype filter. Stratifying "
                         "by the 4 top-level question_types left only 7 items "
                         "in the relational subtypes, which is uninterpretable; "
                         "targeting subtypes directly fixes that.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    cache = outdir / "vol_cache"
    cache.mkdir(parents=True, exist_ok=True)

    rows = load_qa()
    if args.subtypes:
        want = {s.strip() for s in args.subtypes.split(",")}
        rows = [r for r in rows if r["question_subtype"] in want]
        print(f"filtered to subtypes {sorted(want)}: {len(rows)} QA")
    sample = stratified_sample(rows, args.volumes, args.per_volume, args.seed)
    wanted = sorted({r["image_id"] for r in sample})
    print(f"sampled {len(sample)} QA over {len(wanted)} volumes")
    print("by type:", dict(Counter(r["question_type"] for r in sample)))

    ok = []
    for i, vid in enumerate(wanted, 1):
        p = fetch_and_cache(vid, cache, args.keep_source)
        if p:
            ok.append(vid)
        if i % 10 == 0 or i == len(wanted):
            import shutil
            free_gb = shutil.disk_usage("/home").free / 1e9
            print(f"  [{i}/{len(wanted)}] cached={len(ok)}  free={free_gb:.0f}GB",
                  flush=True)

    kept = [r for r in sample if r["image_id"] in set(ok)]
    with open(outdir / "qa.jsonl", "w") as f:
        for r in kept:
            stem, opts = parse_options(r["mc_question"])
            f.write(json.dumps({
                "qid": r["qid"], "image_id": r["image_id"],
                "question": stem, "options": opts,
                "correct_option": r["correct_option"],
                "answer": opts.get(r["correct_option"], r["answer"]),
                "question_type": r["question_type"],
                "question_subtype": r["question_subtype"],
                "requires_tools": r["requires_tools"],
            }) + "\n")
    print(f"\nwrote {len(kept)} QA over {len(ok)} volumes to {outdir/'qa.jsonl'}")


def selftest() -> None:
    stem, opts = parse_options(
        "Find which kidney is larger in volume (or if they're equal): "
        "A: left kidney B: the same C: right kidney")
    assert stem.endswith("equal):"), stem
    assert opts == {"A": "left kidney", "B": "the same",
                    "C": "right kidney"}, opts

    # option text containing a capital letter must not split the string
    s2, o2 = parse_options("Where is the lesion? A: Right lobe B: Left lobe")
    assert o2 == {"A": "Right lobe", "B": "Left lobe"}, o2

    rows = [{"image_id": f"v{i%4}", "question_type": t, "qid": f"q{i}_{t}"}
            for i in range(40) for t in ("recognition", "visual reasoning")]
    s = stratified_sample(rows, n_volumes=3, per_volume=4)
    vols = {r["image_id"] for r in s}
    types = {r["question_type"] for r in s}
    assert len(vols) == 3, vols
    assert types == {"recognition", "visual reasoning"}, types

    print(f"selftest OK — option parsing keeps capitalised text intact "
          f"({o2}), sampling spreads over {len(vols)} volumes and both types")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
