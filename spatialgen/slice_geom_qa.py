"""
Geometric QA answerable from a SINGLE 2D axial slice.

Fills the empty cell of the 2x2
-------------------------------
With Qwen2.5-VL held fixed we have measured:

                     findings questions      geometric questions
    2D native        VQA-RAD  +12.7 pp       <- this file
    3D volumetric    DeepTumorVQA (running)  our 3D set  -5.7 pp

Without the 2D/geometric cell, "the image stops helping in 3D" and "the image
stops helping on geometry" are indistinguishable. Here the anatomy, the question
family and the model are all held fixed; only the input dimensionality changes.

Only relations decidable WITHIN the plane are emitted -- left/right and
anterior/posterior, never superior/inferior, which a single axial slice cannot
show. Asking an unanswerable question would measure the harness, not the model.
Structures are also required to be reasonably present in the slice, since a
few stray voxels give a meaningless centroid.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import label_map  # noqa: E402
from scene_graph import load_ras  # noqa: E402

# In RAS+ a slice at fixed z spans (x=right, y=anterior). Only these two axes
# carry information in-plane.
IN_PLANE = {"lateral": (0, "right", "left"),
            "anteroposterior": (1, "anterior", "posterior")}


@dataclass
class SliceQA:
    qid: str
    category: str
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    slice_index: int = 0
    provenance: dict = field(default_factory=dict)


def pick_slice(seg: np.ndarray, min_structures: int = 6) -> int | None:
    """Axial slice showing the most distinct structures."""
    best, best_n = None, 0
    for z in range(seg.shape[2]):
        n = len(np.unique(seg[:, :, z])) - 1
        if n > best_n:
            best, best_n = z, n
    return best if best_n >= min_structures else None


def slice_structures(seg: np.ndarray, z: int, names: dict[int, str],
                     min_px: int = 150) -> dict[str, dict]:
    """Per-structure 2D centroid and extent within one slice."""
    plane = seg[:, :, z]
    out = {}
    for lab in np.unique(plane):
        if lab == 0:
            continue
        name = names.get(int(lab))
        if not name:
            continue
        m = plane == lab
        n = int(m.sum())
        if n < min_px:
            continue                       # a handful of voxels has no meaningful centre
        idx = np.argwhere(m)
        out[name] = {"n_px": n,
                     "centroid": idx.mean(axis=0).tolist(),
                     "lo": idx.min(axis=0).tolist(),
                     "hi": idx.max(axis=0).tolist()}
    return out


def generate(seg_path: str, vid: str, spacing_mm: float = 1.0,
             min_margin_px: float = 8.0, max_per_cat: int = 12,
             min_structures: int = 6,
             seed: int = 0) -> tuple[list[SliceQA], int | None]:
    seg, affine = load_ras(seg_path)
    z = pick_slice(seg, min_structures=min_structures)
    if z is None:
        return [], None

    names = label_map()
    st = slice_structures(seg, z, names)
    if len(st) < 4:
        return [], z

    rng = random.Random(seed)
    out: list[SliceQA] = []
    keys = sorted(st)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            for axis_key, (ax, pos, neg) in IN_PLANE.items():
                a_lo, a_hi = st[a]["lo"][ax], st[a]["hi"][ax]
                b_lo, b_hi = st[b]["lo"][ax], st[b]["hi"][ax]
                if a_lo > b_hi:
                    margin, ans = a_lo - b_hi, pos
                elif b_lo > a_hi:
                    margin, ans = b_lo - a_hi, neg
                else:
                    continue               # overlapping in-plane: undecidable
                if margin < min_margin_px:
                    continue
                opp = neg if ans == pos else pos
                out.append(SliceQA(
                    qid=f"{vid}_z{z}_{a}__{b}_{axis_key}",
                    category=axis_key,
                    question=(f"In this axial CT slice, is the "
                              f"{a.replace('_', ' ')} located {ans} or {opp} "
                              f"to the {b.replace('_', ' ')}? "
                              f"Answer with one word."),
                    answer=ans, choices=sorted([ans, opp]), slice_index=z,
                    provenance={"subject": a, "object": b, "axis": axis_key,
                                "margin_px": float(margin),
                                "rule": "in-plane bbox separation"},
                ))

    # balance labels within each category so chance is exactly 50%
    by_cat: dict[str, list[SliceQA]] = {}
    for q in out:
        by_cat.setdefault(q.category, []).append(q)
    balanced: list[SliceQA] = []
    for cat, items in by_cat.items():
        groups: dict[str, list[SliceQA]] = {}
        for q in items:
            groups.setdefault(q.answer, []).append(q)
        if len(groups) < 2:
            continue
        k = min(len(v) for v in groups.values())
        k = min(k, max_per_cat // 2)
        for v in groups.values():
            rng.shuffle(v)
            balanced.extend(v[:k])
    rng.shuffle(balanced)
    return balanced, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    segs = sorted(Path(args.seg_cache).glob("*_seg.nii.gz"))
    if args.limit:
        segs = segs[: args.limit]
    outdir = Path(args.outdir)
    (outdir / "qa").mkdir(parents=True, exist_ok=True)

    total = 0
    slices = {}
    for s in segs:
        vid = s.name.replace("_seg.nii.gz", "")
        qs, z = generate(str(s), vid)
        if not qs:
            print(f"  {vid}: no usable slice", flush=True)
            continue
        slices[vid] = z
        with open(outdir / "qa" / f"{vid}.jsonl", "w") as f:
            for q in qs:
                f.write(json.dumps(asdict(q)) + "\n")
        total += len(qs)
        print(f"  {vid}: slice z={z}, {len(qs)} QA "
              f"{dict(Counter(q.category for q in qs))}", flush=True)

    with open(outdir / "slices.json", "w") as f:
        json.dump(slices, f, indent=1)
    print(f"\n{total} QA over {len(slices)} volumes")


def selftest() -> None:
    """Phantom checks the slice machinery; a real shard checks the invariants.

    A hand-built phantom cannot easily produce BOTH labels in BOTH categories,
    and the balancing step correctly discards any category that has only one --
    so a phantom-only test reports an empty set and tells us nothing. Structural
    properties are therefore verified on generated output when it exists.
    """
    seg = np.zeros((60, 60, 10), dtype=np.int16)
    seg[40:55, 20:40, 5] = 1
    seg[5:20, 20:40, 5] = 2
    seg[20:40, 40:55, 5] = 3
    seg[20:40, 2:15, 5] = 4

    assert pick_slice(seg, min_structures=4) == 5
    assert pick_slice(seg, min_structures=99) is None, "must refuse a thin slice"
    st = slice_structures(seg, 5, label_map())
    assert len(st) == 4, list(st)
    # a structure present as a speck must be dropped, not given a centroid
    seg[0, 0, 5] = 5
    assert len(slice_structures(seg, 5, label_map())) == 4

    shard = Path(__file__).resolve().parent.parent / "slice2d" / "qa"
    files = sorted(shard.glob("*.jsonl")) if shard.is_dir() else []
    if not files:
        print("selftest OK — slice selection and structure filtering verified "
              "(no generated shard present for invariant checks)")
        return

    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    cats = Counter(r["category"] for r in rows)
    # a single axial slice cannot decide superior/inferior
    assert "longitudinal" not in cats, cats
    assert set(cats) <= {"lateral", "anteroposterior"}, cats

    for cat in cats:
        lab = Counter(r["answer"] for r in rows if r["category"] == cat)
        assert len(set(lab.values())) == 1, (cat, lab)

    for r in rows:
        assert r["answer"] in r["choices"], r
        assert r["provenance"]["margin_px"] >= 8.0, r
        assert r["provenance"]["axis"] == r["category"], r

    print(f"selftest OK — slice machinery verified on a phantom; "
          f"{len(rows)} generated QA over {len(files)} volumes carry only "
          f"in-plane categories {dict(cats)}, balanced per category, every "
          f"answer within its choices and above the margin floor")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
