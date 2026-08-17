"""
Machine-generated inventory of every 3D result file on disk.

Purpose: the Drive archive accumulated 243 top-level *.jsonl result files across
several months of runs, with no manifest. Before any of it can be cited in a
journal submission we need to know, per file: which experiment arm it is, how
many probes it holds, whether gold labels are present, and what its raw accuracy
is. Nothing here interprets; it only reports what each file contains.

Naming convention, recovered from the run scripts in spatialgen/:

  mm_<model>_<arm>            main multi-model run, arm in {sighted, blind}
  sc_<organ>_<model>_<arm>    single-organ run
  roi_<organ>_<model>_<arm>   ROI ablation, arm in {full, zero, roi_only,
                              roi_masked} with optional _air/_local fill suffix
  fam_/tv_/v3_/v4_/tp_/lpa_   trap and probe families (cal suffix = calibration)
  calib_<model>_<arm>         content-free calibration baseline
  decay_/sanity_/blind_       ablations and controls
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "from3d")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

ORGANS = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task10_Colon"]
ARMS = ["sighted", "blind", "full", "zero", "roi_only", "roi_masked",
        "roi_only_air", "roi_masked_air", "roi_only_local", "roi_masked_local"]

# prefix -> what the family measures, from the run scripts and paper sections
FAMILY = {
    "mm": "main multi-model audit (sighted vs blind)",
    "sc": "single-organ run",
    "roi": "ROI four-arm grounding ablation",
    "fam": "trap family 'fam' (600 probes)",
    "tv": "trap family 'tv' (226 probes)",
    "tvcal": "trap family 'tv' content-free calibration",
    "v3": "trap family 'v3' (77 probes)",
    "v3cal": "trap family 'v3' content-free calibration",
    "v4": "trap family 'v4' (600 probes)",
    "v4cal": "trap family 'v4' content-free calibration",
    "tp": "target-pair differentiation",
    "lpa": "language-prior accuracy",
    "lpacal": "language-prior accuracy calibration",
    "calib": "content-free calibration baseline",
    "decay": "visual information decay sweep",
    "sanity": "sanity / response-channel control",
    "blind": "blind-input control",
    "m3d": "M3D standalone run",
}


def parse_name(stem: str) -> dict:
    """Recover (family, organ, model, arm) from a result filename stem."""
    organ = next((o for o in ORGANS if o in stem), "")
    rest = stem.replace(organ + "_", "") if organ else stem
    parts = rest.split("_")
    family = parts[0]
    # longest matching arm suffix wins, so roi_masked_air beats roi_masked
    arm = ""
    for a in sorted(ARMS, key=len, reverse=True):
        if rest.endswith("_" + a) or rest == a:
            arm = a
            rest = rest[: -(len(a) + 1)]
            break
    model = "_".join(rest.split("_")[1:]) if "_" in rest else ""
    return {"family": family, "organ": organ, "model": model, "arm": arm}


def scan(path: str) -> dict:
    """Read one jsonl result file and summarise it without interpreting it."""
    n = n_gold = n_correct = n_logprob = 0
    preds: Counter = Counter()
    organs: Counter = Counter()
    bad = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n += 1
            p, g = r.get("prediction"), r.get("gold")
            if p is not None:
                preds[str(p)[:24]] += 1
            if g is not None:
                n_gold += 1
                if str(p).strip().lower() == str(g).strip().lower():
                    n_correct += 1
            if r.get("logprobs"):
                n_logprob += 1
            if r.get("organ"):
                organs[r["organ"]] += 1
    modal, modal_n = preds.most_common(1)[0] if preds else ("", 0)
    return {
        "n": n,
        "n_gold": n_gold,
        "acc": round(100.0 * n_correct / n_gold, 2) if n_gold else "",
        "has_logprobs": "yes" if n_logprob else "no",
        "modal_pred": modal,
        "modal_share": round(100.0 * modal_n / n, 1) if n else "",
        "organs_in_file": ";".join(sorted(organs)),
        "malformed_lines": bad,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(ROOT)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(ROOT, fn)
        meta = parse_name(fn[: -len(".jsonl")])
        row = {"file": fn, **meta,
               "family_meaning": FAMILY.get(meta["family"], "unclassified"),
               "size_kb": round(os.path.getsize(path) / 1024, 1),
               **scan(path)}
        rows.append(row)

    cols = ["file", "family", "family_meaning", "organ", "model", "arm", "n",
            "n_gold", "acc", "has_logprobs", "modal_pred", "modal_share",
            "organs_in_file", "malformed_lines", "size_kb"]
    dest = os.path.join(OUT, "results_index.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    fam = Counter(r["family"] for r in rows)
    unclassified = [r["file"] for r in rows if r["family_meaning"] == "unclassified"]
    empty = [r["file"] for r in rows if r["n"] == 0]
    nogold = [r["file"] for r in rows if r["n"] and not r["n_gold"]]
    degenerate = [r["file"] for r in rows
                  if isinstance(r["modal_share"], float) and r["modal_share"] >= 99.0]

    print(f"indexed {len(rows)} result files -> {dest}")
    print(f"total records: {sum(r['n'] for r in rows):,}")
    print("\nby family:")
    for k, v in fam.most_common():
        print(f"  {k:10s} {v:4d}  {FAMILY.get(k, 'unclassified')}")
    print(f"\nunclassified files ({len(unclassified)}): {unclassified[:10]}")
    print(f"empty files ({len(empty)}): {empty[:10]}")
    print(f"files with no gold labels ({len(nogold)}): {nogold[:10]}")
    print(f"\nfiles where one answer is >=99% of predictions ({len(degenerate)}):")
    for f in degenerate:
        r = next(x for x in rows if x["file"] == f)
        print(f"  {f:58s} modal={r['modal_pred']!r:8s} {r['modal_share']}%")


if __name__ == "__main__":
    main()
