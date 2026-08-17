"""
Build the canonical result tables for the journal version.

Everything here is derived from raw files on disk, never transcribed from the
drafts. Each table carries a `provenance` column naming the files it came from
and an `attested` column saying whether the repository's own analysis script
will still certify it today. That second column is the point of this script:
several numbers the drafts quote are frozen in figdata/*.csv and can no longer
be regenerated, and a journal submission has to know which ones those are.

Outputs land in out/tables/.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = os.path.join(BASE, "from3d")
FIG = os.path.join(D3, "figdata")
REF2D = os.path.join(D3, "paper", "ref2d")
OUT = os.path.join(BASE, "out", "tables")

ORDER = ["smolvlm", "qwen3b", "qwen7b", "internvl", "qwen3vl", "internvl14",
         "qwen32b", "llavaov", "idefics3", "pixtral", "aria",
         "m3d", "m3dllama", "med3dvlm"]
LABEL = {"smolvlm": "SmolVLM2-2.2B", "qwen3b": "Qwen2.5-VL-3B",
         "qwen7b": "Qwen2.5-VL-7B", "internvl": "InternVL3-8B",
         "qwen3vl": "Qwen3-VL-8B", "internvl14": "InternVL3-14B",
         "qwen32b": "Qwen2.5-VL-32B", "llavaov": "LLaVA-OneVision-7B",
         "idefics3": "Idefics3-8B", "pixtral": "Pixtral-12B",
         "aria": "Aria-25B-MoE", "m3d": "M3D-LaMed-Phi3-4B",
         "m3dllama": "M3D-LaMed-Llama2-7B", "med3dvlm": "Med3DVLM-7B"}
NATIVE = {"m3d", "m3dllama", "med3dvlm"}


def load(path):
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["qid"]] = r
    return out


def write(name, rows, cols):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name:38s} {len(rows):4d} rows")


def table_main_audit():
    """Sighted vs blind on the common subset, recomputed from mm_*.jsonl."""
    arms = {}
    for m in ORDER:
        s, b = load(f"{D3}/mm_{m}_sighted.jsonl"), load(f"{D3}/mm_{m}_blind.jsonl")
        if s and b:
            arms[m] = (s, b)
    if not arms:
        return []
    common = set.intersection(*[set(s) & set(b) for s, b in arms.values()])
    common = sorted(common)
    golds = [arms[next(iter(arms))][0][q]["gold"] for q in common]
    rows = []
    for m in ORDER:
        if m not in arms:
            rows.append({"model": LABEL[m], "input": "native" if m in NATIVE else "montage",
                         "n": "", "sighted_acc": "", "blind_acc": "", "image_gain": "",
                         "status": "RUN INCOMPLETE", "attested": "no",
                         "provenance": f"mm_{m}_{{sighted,blind}}.jsonl absent"})
            continue
        s, b = arms[m]
        sa = 100.0 * sum(s[q]["prediction"] == s[q]["gold"] for q in common) / len(common)
        ba = 100.0 * sum(b[q]["prediction"] == b[q]["gold"] for q in common) / len(common)
        rows.append({
            "model": LABEL[m], "input": "native" if m in NATIVE else "montage",
            "n": len(common), "sighted_acc": round(sa, 1), "blind_acc": round(ba, 1),
            "image_gain": f"{sa - ba:+.1f}", "status": "ok", "attested": "yes",
            "provenance": f"mm_{m}_sighted.jsonl, mm_{m}_blind.jsonl",
        })
    print(f"    common subset n={len(common)} "
          f"({golds.count('yes')} yes / {golds.count('no')} no)")
    return rows


def table_confound_free():
    """Frozen forest-plot data: the growth-matched, confound-free subset."""
    rows = []
    for r in csv.DictReader(open(f"{FIG}/fig7_forest_confound_free.csv")):
        ci_lo, ci_hi = float(r["ci_lo"]), float(r["ci_hi"])
        rows.append({
            "model": r["model"], "input": r["input"], "acc": r["acc"],
            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "blind": r["blind"],
            "image_gain": r["image_gain"],
            "ci_contains_50": "yes" if ci_lo <= 50.0 <= ci_hi else "no",
            "modal_share": r["modal_share"], "modal_token": r["modal_token"],
            "attested": "frozen-csv-only",
            "provenance": "figdata/fig7_forest_confound_free.csv",
        })
    accs = [float(r["acc"]) for r in rows]
    gains = [float(r["image_gain"]) for r in rows]
    n50 = sum(r["ci_contains_50"] == "yes" for r in rows)
    print(f"    acc {min(accs):.1f}-{max(accs):.1f} %, gain {min(gains):+.1f} to "
          f"{max(gains):+.1f} pp, {n50}/{len(rows)} CIs contain 50")
    return rows


def table_response_controls():
    """The known-answer control: can the model answer 'Is this a CT scan?'."""
    rows = []
    for r in csv.DictReader(open(f"{FIG}/fig8_response_controls.csv")):
        pr = float(r["pass_rate"])
        rows.append({
            "model": r["model"], "input": r["input"],
            "perfect_questions": r["perfect_questions"], "n_questions": r["n_questions"],
            "pass_rate": r["pass_rate"], "yes_rate": r["yes_rate"],
            "channel_verdict": "FAILS - not scoreable" if pr < 60 else "ok",
            "attested": "frozen-csv-only",
            "provenance": "figdata/fig8_response_controls.csv",
        })
    bad = [r["model"] for r in rows if r["channel_verdict"].startswith("FAILS")]
    print(f"    {len(bad)}/{len(rows)} models fail the response-channel control: "
          f"{', '.join(bad)}")
    return rows


def table_four_arm():
    """ROI four-arm. Frozen only -- the repo's own guard now refuses to attest."""
    guard = os.path.join(BASE, "out", "analyse_roi_4arm.txt")
    verdict = "frozen-csv-only; analyse_roi_4arm.py REFUSES (fill provenance lost)"
    if os.path.exists(guard) and "not attestable" not in open(guard).read():
        verdict = "regenerated"
    rows = []
    for r in csv.DictReader(open(f"{FIG}/fig9_roi_four_arm.csv")):
        rows.append({**r, "attested": verdict,
                     "provenance": "figdata/fig9_roi_four_arm.csv"})
    # the contrast the argument rests on: does masking the evidence cost anything?
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["organ"])][r["arm"]] = float(r["acc"])
    deltas = [c["full"] - c["roi_masked"] for c in by.values()
              if "full" in c and "roi_masked" in c]
    if deltas:
        print(f"    cost of masking the evidence region across {len(deltas)} "
              f"model-organ cells: {min(deltas):+.1f} to {max(deltas):+.1f} pp")
    return rows


def table_2d_reference():
    """The conference-version 2D metrics, for the cross-modality comparison."""
    rows = []
    for r in csv.DictReader(open(f"{REF2D}/metrics2d.csv")):
        vgr = float(r["vgr"]) * 100
        rows.append({
            "model": r["model"], "n_total": r["n_total"],
            "overall_acc": round(float(r["overall_acc"]) * 100, 1),
            "sfr": round(float(r["sfr"]) * 100, 1),
            "vgr_pp": round(vgr, 1), "vgr_sign": "positive" if vgr >= 0 else "NEGATIVE",
            "attested": "yes", "provenance": "from3d/paper/ref2d/metrics2d.csv",
        })
    neg = [r for r in rows if r["vgr_sign"] == "NEGATIVE"]
    pos = [r["vgr_pp"] for r in rows if r["vgr_sign"] == "positive"]
    print(f"    {len(rows)} models; VGR negative for {len(neg)} "
          f"({', '.join(r['model'] for r in neg)})")
    print(f"    positive VGR spans {min(pos):+.1f} to {max(pos):+.1f} pp")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Table 1  3D main audit (recomputed from raw)")
    write("t1_3d_main_audit.csv", table_main_audit(),
          ["model", "input", "n", "sighted_acc", "blind_acc", "image_gain",
           "status", "attested", "provenance"])
    print("Table 2  confound-free growth-matched subset")
    write("t2_confound_free.csv", table_confound_free(),
          ["model", "input", "acc", "ci_lo", "ci_hi", "blind", "image_gain",
           "ci_contains_50", "modal_share", "modal_token", "attested", "provenance"])
    print("Table 3  response-channel controls")
    write("t3_response_controls.csv", table_response_controls(),
          ["model", "input", "perfect_questions", "n_questions", "pass_rate",
           "yes_rate", "channel_verdict", "attested", "provenance"])
    print("Table 4  ROI four-arm")
    write("t4_roi_four_arm.csv", table_four_arm(),
          ["model", "organ", "arm", "n", "acc", "ci_lo", "ci_hi", "modal_share",
           "attested", "provenance"])
    print("Table 5  2D conference reference metrics")
    write("t5_2d_reference.csv", table_2d_reference(),
          ["model", "n_total", "overall_acc", "sfr", "vgr_pp", "vgr_sign",
           "attested", "provenance"])
    print(f"\nall tables in {OUT}")


if __name__ == "__main__":
    main()
