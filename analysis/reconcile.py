"""
Reconcile the frozen figure data against a fresh re-run of the analysis scripts.

The figdata/*.csv files are the numbers the drafts quote. They were produced on
the original machine months ago. This checks whether the scripts still in the
repository, run against the data still on disk, reproduce them.

A mismatch is not automatically an error -- it can mean the CSV was frozen from a
different run of the same arm -- but every mismatch has to be resolved before a
number is quoted in a submission.
"""
from __future__ import annotations

import csv
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "from3d", "figdata")
OUT = os.path.join(BASE, "out")

TOL = 0.15  # percentage points / nats; frozen CSVs are rounded to 3 decimals


def read_csv(name):
    with open(os.path.join(FIG, name)) as fh:
        return list(csv.DictReader(fh))


def parse_margin_rerun():
    """Pull the margin table out of analyse_margin.py's stdout."""
    path = os.path.join(OUT, "analyse_margin.txt")
    if not os.path.exists(path):
        return {}
    rows = {}
    for line in open(path):
        m = re.match(
            r"^(\S.*?)\s+(montage|native)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+\[",
            line.rstrip())
        if m:
            rows[m.group(1).strip()] = {
                "input": m.group(2),
                "perturbation": float(m.group(3)),
                "gap": float(m.group(4)),
                "flip_rate": float(m.group(5)),
            }
    return rows


def cmp_num(a, b, tol=TOL):
    try:
        return "match" if abs(float(a) - float(b)) <= tol else "MISMATCH"
    except (TypeError, ValueError):
        return "n/a"


def main():
    print("=" * 78)
    print("fig10_margin.csv  vs  fresh analyse_margin.py")
    print("=" * 78)
    frozen = {r["model"]: r for r in read_csv("fig10_margin.csv")}
    rerun = parse_margin_rerun()
    print(f"{'model':22s} {'field':13s} {'frozen':>9s} {'rerun':>9s}  verdict")
    n_mis = 0
    for model, fr in frozen.items():
        rr = rerun.get(model)
        if not rr:
            print(f"{model:22s} {'-':13s} {'':>9s} {'':>9s}  NOT IN RERUN")
            continue
        for field in ("perturbation", "gap", "flip_rate"):
            v = cmp_num(fr[field], rr[field])
            if v == "MISMATCH":
                n_mis += 1
                print(f"{model:22s} {field:13s} {float(fr[field]):9.3f} "
                      f"{rr[field]:9.3f}  {v}")
    print(f"\n{n_mis} mismatching cells")

    print()
    print("=" * 78)
    print("fig7_forest_confound_free.csv  ->  ranges the drafts quote")
    print("=" * 78)
    rows = read_csv("fig7_forest_confound_free.csv")
    accs = [float(r["acc"]) for r in rows]
    gains = [float(r["image_gain"]) for r in rows]
    contains50 = [50.0 >= float(r["ci_lo"]) and 50.0 <= float(r["ci_hi"]) for r in rows]
    print(f"n models              : {len(rows)}")
    print(f"accuracy range        : {min(accs):.1f} - {max(accs):.1f} %")
    print(f"image-gain range      : {min(gains):+.1f} to {max(gains):+.1f} pp")
    print(f"CIs containing 50     : {sum(contains50)}/{len(rows)}")
    print()
    print("draft claims to check against the above:")
    print("  PAPER.md abstract   : '48.7-51.0 %' and '-1.3 to +1.0 pp'")
    print("  PAPER.md contrib #2 : '48.0-50.6 %' and '-3.1 to +0.7 pp'")
    print("  -> these two statements are in the SAME document and disagree.")
    lo, hi, glo, ghi = min(accs), max(accs), min(gains), max(gains)
    for label, a, b, c, d in [
        ("abstract", 48.7, 51.0, -1.3, 1.0),
        ("contrib #2", 48.0, 50.6, -3.1, 0.7),
    ]:
        ok = (abs(a - lo) < .05 and abs(b - hi) < .05
              and abs(c - glo) < .05 and abs(d - ghi) < .05)
        print(f"  {label:11s} -> {'CONSISTENT with data' if ok else 'DOES NOT MATCH data'}")
    print(f"  data says           : {lo:.1f}-{hi:.1f} % and {glo:+.1f} to {ghi:+.1f} pp")

    print()
    print("=" * 78)
    print("fig9_roi_four_arm.csv  ->  can analyse_roi_4arm.py still attest it?")
    print("=" * 78)
    rows = read_csv("fig9_roi_four_arm.csv")
    models = sorted({r["model"] for r in rows})
    print(f"frozen CSV covers {len(models)} models: {', '.join(models)}")
    guard = os.path.join(OUT, "analyse_roi_4arm.txt")
    if os.path.exists(guard):
        print("fresh run says:")
        for line in open(guard):
            print("  " + line.rstrip())
    print("the four-arm contrast the draft argues from, per organ:")
    for model in models:
        for organ in ["Lung", "Colon", "Pancreas", "Liver"]:
            cells = {r["arm"]: r for r in rows
                     if r["model"] == model and r["organ"] == organ}
            if len(cells) < 4:
                continue
            full = float(cells["full"]["acc"])
            masked = float(cells["roi_masked"]["acc"])
            only = float(cells["roi_only"]["acc"])
            zero = float(cells["zero"]["acc"])
            print(f"  {model:18s} {organ:9s} full {full:5.1f}  roi_masked {masked:5.1f}"
                  f"  (cost of masking evidence {full - masked:+5.1f} pp)   "
                  f"roi_only {only:5.1f}  zero {zero:5.1f}")


if __name__ == "__main__":
    main()
