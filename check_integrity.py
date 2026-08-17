"""
Sweep every result file for silent truncation and unmatched comparisons.

Two failure modes this catches, both of which occurred:

  truncation -- a run that hit CUDA OOM on some items still writes the rest and
    exits 0. The queue's old resume guard accepted any non-empty file, so a
    446-of-600 arm would have entered a VGR unnoticed.

  arm mismatch -- VGR, image gain and pair consistency are differences between
    two arms over *identical* items. If one arm is short, the difference is
    taken over different item sets and is not interpretable, however close the
    row counts look.

Expected counts are derived from the data rather than hard-coded, so the check
stays honest if the corpora change.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import re
from collections import defaultdict

import os as _os
R = _os.environ.get("MEDVIGIL3D_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
def being_written(path: str) -> bool:
    """Is a runner currently writing this file?

    A group can disagree because it is broken or because it is half-written,
    and those need different responses: one is a defect, the other is a clock.
    Reporting them the same way trains you to ignore the check during the very
    hours the runs are happening.
    """
    return subprocess.run(["pgrep", "-af", "run_roi_arms.py|run_multimodel.py|"
                           "calibrate_families.py"],
                          capture_output=True, text=True
                          ).stdout.find(os.path.basename(path)) >= 0


def rows(p: str) -> int:
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def qids(p: str) -> set[str]:
    return {json.loads(l)["qid"] for l in open(p)} if os.path.exists(p) else set()


def report(title: str, groups: dict[str, list[str]]) -> int:
    """Each group is a set of files that must agree item-for-item."""
    bad = 0
    print(f"\n=== {title} ===")
    for key, files in sorted(groups.items()):
        files = [f for f in files if os.path.exists(f)]
        if len(files) < 2:
            if files:
                print(f"  {key:52} {rows(files[0]):5d}  (single arm, nothing to match)")
            continue
        live = [f for f in files if being_written(f)]
        if live:
            print(f"  {key:52} in progress — "
                  + ", ".join(f"{os.path.basename(f).split('_')[-1][:-6]}={rows(f)}"
                              for f in files)
                  + f"  (writing: {os.path.basename(live[0])})")
            continue
        sets = {f: qids(f) for f in files}
        n = {f: len(s) for f, s in sets.items()}
        common = set.intersection(*sets.values())
        agree = all(s == common for s in sets.values())
        flag = "OK " if agree else "MISMATCH"
        if not agree:
            bad += 1
        print(f"  {key:52} {flag} " +
              " ".join(f"{os.path.basename(f).split('_')[-1][:-6]}={n[f]}"
                       for f in files) +
              ("" if agree else f"  common={len(common)}"))
    return bad


def main() -> None:
    bad = 0

    # benchmark arms: sighted vs blind, per model
    g = defaultdict(list)
    for f in glob.glob(R + "mm_*_sighted.jsonl"):
        m = re.match(r".*mm_(.+)_sighted\.jsonl", f)[1]
        g[m] = [R + f"mm_{m}_sighted.jsonl", R + f"mm_{m}_blind.jsonl"]
    bad += report("benchmark arms (sighted vs blind)", g)

    # target-contrast pairs
    g = defaultdict(list)
    for f in glob.glob(R + "tp_*_sighted.jsonl"):
        m = re.match(r".*tp_(.+)_sighted\.jsonl", f)[1]
        g[m] = [R + f"tp_{m}_sighted.jsonl", R + f"tp_{m}_blind.jsonl"]
    bad += report("target-contrast pairs", g)

    # trap families
    g = defaultdict(list)
    for f in glob.glob(R + "t[v3]_*_sighted.jsonl"):
        tag, m = re.match(r".*/(t[v3])_(.+)_sighted\.jsonl", f).groups()
        g[f"{tag} {m}"] = [R + f"{tag}_{m}_sighted.jsonl", R + f"{tag}_{m}_blind.jsonl"]
    bad += report("trap families", g)

    # ROI arms: every condition present for one organ+model must match
    g = defaultdict(list)
    for f in glob.glob(R + "roi_*.jsonl"):
        b = os.path.basename(f)[4:-6]
        for c in ("roi_only", "roi_masked", "full", "zero"):
            if b.endswith("_" + c) or f"_{c}_" in b:
                g[b.replace("_" + c, "").replace(f"_{c}_", "_")].append(f)
                break
    bad += report("ROI arms (all conditions must share items)", g)

    print(f"\n{'ALL GROUPS MATCHED' if not bad else f'{bad} GROUP(S) MISMATCHED'}"
          " — a mismatched group must not enter a difference-based metric")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
