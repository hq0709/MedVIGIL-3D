"""
One command that prints every number from this round of experiments, from the
raw run files, with the caveats attached to the numbers they qualify.

    python runs/summarise.py

Sections
--------
  1. What the model was actually shown   (E1 input audit, from the pixels)
  2. Solvability ceiling                 (E2, plus the framing/decoding controls)
  3. Identification control              (E1)
  4. Sub-task decomposition              (E3)
  5. Native volumetric response channel  (E6, from files already in the repo)

Nothing here recomputes a subset or re-derives a label; it reads the run files
and the committed corpus.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
RESULTS = os.path.join(REPO, "results_new")

MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"),
          ("qwen3vl", "Qwen3-VL-8B"), ("qwen32b", "Qwen2.5-VL-32B"),
          ("qwen72b", "Qwen2.5-VL-72B"), ("geometry", "reference rule")]
IMAGE_CONDS = ["plain", "bestslice", "overlay", "identified"]
TEXT_CONDS = ["geometry-oracle", "numeric-oracle", "numeric-oracle-gen",
              "text-oracle", "text-oracle-gen", "text-oracle-cot",
              "text-oracle-blind"]


def rows_for(model: str, cond: str) -> list[dict]:
    out: dict[str, dict] = {}
    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("id_") or not name.endswith(f"_{model}_{cond}.jsonl"):
            continue
        for line in open(os.path.join(RESULTS, name)):
            if line.strip():
                r = json.loads(line)
                out[r["qid"]] = r
    return list(out.values())


def stats(rows: list[dict]) -> dict:
    by_pair = defaultdict(list)
    for r in rows:
        if r.get("pair_id"):
            by_pair[r["pair_id"]].append(r)
    full = [v for v in by_pair.values() if len(v) == 2]
    sing = [v[0] for v in by_pair.values() if len(v) == 1]
    d = {"n": len(rows),
         "acc": 100.0 * sum(r["prediction"] == r["gold"] for r in rows) / len(rows),
         "pairs": len(full)}
    d["acc_pairs"] = (100.0 * sum(r["prediction"] == r["gold"]
                                  for v in full for r in v) / (2 * len(full))
                      if full else float("nan"))
    d["acc_sing"] = (100.0 * sum(r["prediction"] == r["gold"] for r in sing)
                     / len(sing) if sing else float("nan"))
    d["same"] = (100.0 * sum(v[0]["prediction"] == v[1]["prediction"] for v in full)
                 / len(full) if full else float("nan"))
    c = Counter(r["prediction"] for r in rows)
    d["top"] = f"{c.most_common(1)[0][0]} {100 * c.most_common(1)[0][1] / len(rows):.0f}%"
    return d


def section(title: str) -> None:
    print(f"\n\n{'=' * 100}\n{title}\n{'=' * 100}")


def confound_table() -> None:
    """What a single number can score on each reporting unit, without an image.

    growth_matched.py bins by growth and keeps equal yes/no inside each bin, and
    reports the resulting growth-only ceiling (50.8%) as the reason above-chance
    accuracy on that subset cannot come from the number. That is true of the
    growth number and false of the other one. "Yes" means gap <= growth, so
    inside a narrow growth bin the yes-items are exactly those with the smaller
    gaps: matching on growth makes the GAP predictive. It is not a cue the
    published models could reach, because they were reading almost nothing off
    the montage -- but it is precisely what an annotated image with a scale bar
    hands a model, which is what the identification control set out to give it.

    Complete matched pairs hold gap fixed, so the gap cue is worth exactly
    chance there -- but they do NOT remove every cue, and the table below is
    printed rather than summarised because of it: a single threshold on the
    growth amount scores 83.7% on the pairs. No unit here is clean of both
    numbers. What makes the pair endpoint the right one for THIS experiment is
    narrower and needs saying that way: the identification control's whole
    manipulation is to reveal the gap, and the pairs are where revealing it
    cannot help. The growth amount is in the question text in every arm and
    every condition, so it is held constant across the comparison -- and no
    model goes near 83.7%, which is itself the evidence that none of them is
    thresholding on it.
    """
    import numpy as np
    from growth_matched import growth_of, matched_subset
    ref, prov = {}, {}
    for line in open(os.path.join(REPO, "common_subset", "qa", "all.jsonl")):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
        prov[r["qid"]] = r["provenance"]
    keep = matched_subset(ref, growth_of())
    by_pair = defaultdict(list)
    for q in sorted(keep):
        by_pair["_".join(q.split("_")[:-1])].append(q)
    units = [("full common subset", sorted(ref)),
             ("growth-matched subset", sorted(keep)),
             ("  of which complete pairs",
              [q for v in by_pair.values() if len(v) == 2 for q in v]),
             ("  of which singletons",
              [v[0] for v in by_pair.values() if len(v) == 1])]
    print(f"{'reporting unit':28}{'n':>7}{'growth-only':>14}{'gap-only':>11}")
    print("-" * 60)
    for name, sel in units:
        g = np.array([prov[q]["gap_mm"] for q in sel])
        w = np.array([prov[q]["growth_mm"] for q in sel])
        y = np.array([ref[q] == "yes" for q in sel])
        bg = max(((g <= t) == y).mean() for t in np.unique(g))
        bw = max(((w > t) == y).mean() for t in np.unique(w))
        print(f"{name:28}{len(sel):>7}{100 * bw:>13.1f}%{100 * bg:>10.1f}%")
    print("\nBest single-threshold rule on that number alone, threshold chosen "
          "on the same items, so both\ncolumns are upper bounds. Matching on "
          "growth removed the growth cue and created a gap cue.")


def main() -> None:
    section("0. WHAT A SINGLE NUMBER CAN SCORE -- confounds per reporting unit")
    confound_table()

    section("1. WHAT THE MODEL WAS SHOWN -- input audit of the four conditions "
            "(pixels, no model)")
    path = os.path.join(RESULTS, "condition_audit.jsonl")
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        print(f"{'condition':12}{'n':>6}{'lesion shown':>14}{'target shown':>14}"
              f"{'both':>8}{'red drawn':>11}{'cyan drawn':>12}")
        print("-" * 77)
        for cond in IMAGE_CONDS:
            n = len(rows)
            f = lambda k: 100.0 * sum(r[cond][k] > 0 for r in rows) / n
            print(f"{cond:12}{n:>6}{f('lesion_voxels_shown'):>13.1f}%"
                  f"{f('target_voxels_shown'):>13.1f}%"
                  f"{100.0 * sum(r[cond]['lesion_voxels_shown'] > 0 and r[cond]['target_voxels_shown'] > 0 for r in rows) / n:>7.1f}%"
                  f"{f('lesion_outline_px'):>10.1f}%{f('target_outline_px'):>11.1f}%")
        print("\n`plain` is the published condition. The lesion the question "
              "names is absent from every slice shown for 42% of probes.\n"
              "`overlay` promises a red outline in its legend for 100% of "
              "probes and draws one for 58%.")
    else:
        print("  (not run: runs/audit_conditions.py)")

    section("2. SOLVABILITY CEILING (E2) -- same probes, no image except where "
            "stated")
    print(f"{'model':18}{'arm':21}{'n':>6}{'acc':>8}{'pairs':>7}"
          f"{'acc|pairs':>11}{'same':>7}   modal answer")
    print("-" * 100)
    for tag, name in MODELS:
        for cond in TEXT_CONDS:
            rows = rows_for(tag, cond)
            if not rows:
                continue
            s = stats(rows)
            print(f"{name:18}{cond:21}{s['n']:>6}{s['acc']:>7.1f}%{s['pairs']:>7}"
                  f"{s['acc_pairs']:>10.1f}%{s['same']:>6.0f}%   {s['top']}")
    print("\n`numeric-oracle` is the same comparison with the clinical wording "
          "removed. Read it against `text-oracle`:\nthe arithmetic is available, "
          "and the wording is what removes it. `acc|pairs` is the endpoint that\n"
          "cannot be reached through gap magnitude.")

    section("3. IDENTIFICATION CONTROL (E1) -- growth-matched subset, four organs")
    print(f"{'model':18}{'condition':12}{'n':>6}{'acc':>8}{'pairs':>7}"
          f"{'acc|pairs':>11}{'acc|singl.':>12}{'same':>7}{'yes-rate':>10}")
    print("-" * 101)
    for tag, name in MODELS:
        any_row = False
        for cond in IMAGE_CONDS:
            rows = rows_for(tag, cond)
            if not rows:
                continue
            any_row = True
            s = stats(rows)
            yr = 100.0 * sum(r["prediction"] == "yes" for r in rows) / len(rows)
            print(f"{name:18}{cond:12}{s['n']:>6}{s['acc']:>7.1f}%{s['pairs']:>7}"
                  f"{s['acc_pairs']:>10.1f}%{s['acc_sing']:>11.1f}%"
                  f"{s['same']:>6.0f}%{yr:>9.1f}%")
        if any_row:
            print()
    print("Intervals: analysis/identification_control_ci.py (volume-clustered "
          "bootstrap, B=10,000).\n`acc|singl.` is accuracy on probes whose "
          "matched partner the growth matching dropped; gap correlates with the\n"
          "label there, so a gain confined to that column is a proximity cue "
          "rather than the counterfactual.")

    section("3b. `overlay` split by whether the red outline was actually drawn")
    print("The legend sentence is added per condition, so in `overlay` every "
          "probe is told the lesion is\noutlined in red while 42% of them show "
          "no red pixel. Splitting on the recorded geometry turns that\nflaw "
          "into the comparison it accidentally is: same condition, same legend, "
          "annotation present or not.\n")
    print(f"{'model':18}{'red outline':14}{'n':>6}{'acc':>8}{'pairs':>7}"
          f"{'acc|pairs':>11}{'same':>7}")
    print("-" * 71)
    for tag, name in MODELS:
        rows = rows_for(tag, "overlay")
        if not rows:
            continue
        for label, sel in (("drawn", True), ("absent", False)):
            sub = [r for r in rows
                   if bool(r.get("geometry", {}).get("lesion_outline_px", 0)) is sel]
            if not sub:
                continue
            s = stats(sub)
            print(f"{name:18}{label:14}{s['n']:>6}{s['acc']:>7.1f}%{s['pairs']:>7}"
                  f"{s['acc_pairs']:>10.1f}%{s['same']:>6.0f}%")

    section("4. SUB-TASK DECOMPOSITION (E3) -- `identified` rendering, four-way, "
            "chance 25%")
    print(f"{'model':18}{'sub-task':20}{'n':>6}{'acc':>8}   modal answer")
    print("-" * 70)
    for tag, name in MODELS:
        for st in ["localise", "name", "distance"]:
            rows = rows_for(tag, f"subtask-{st}")
            if not rows:
                continue
            s = stats(rows)
            print(f"{name:18}{st:20}{s['n']:>6}{s['acc']:>7.1f}%   {s['top']}")

    section("5. NATIVE VOLUMETRIC MODELS (E6) -- from files already in the repo")
    print(f"{'model':22}{'controls native':>17}{'controls montage':>18}"
          f"{'recovery':>10}")
    print("-" * 67)
    for tag, name in [("m3d", "M3D-LaMed-Phi3-4B"),
                      ("m3dllama", "M3D-LaMed-Llama2-7B"),
                      ("med3dvlm", "Med3DVLM-7B")]:
        try:
            a = [json.loads(l) for l in open(os.path.join(REPO, f"sanity_{tag}.jsonl"))]
            b = [json.loads(l) for l in
                 open(os.path.join(REPO, f"sanity_{tag}_montageprompt.jsonl"))]
        except FileNotFoundError:
            continue
        fa = 100.0 * sum(r["prediction"] == r["gold"] for r in a) / len(a)
        fb = 100.0 * sum(r["prediction"] == r["gold"] for r in b) / len(b)
        print(f"{name:22}{fa:>16.1f}%{fb:>17.1f}%{fa - fb:>+9.1f}")
    print("Montage-input models score 100% on the same controls. Native framing "
          "recovers M3D-LaMed-Phi3 by\n29.3 points, not the 41 quoted in "
          "run_multimodel.py; Med3DVLM answers \"no\" to all 140 controls under "
          "both\nframings. On the matched subset all three stay at chance under "
          "either framing.")


if __name__ == "__main__":
    main()
