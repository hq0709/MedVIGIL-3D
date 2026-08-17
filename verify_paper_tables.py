"""
Check the paper's headline table against the files it claims to summarise.

Three defects found in one day -- a metric whose null was borrowed from a
degenerate control arm, ROI numbers whose fill could not be recovered, and a
reproduction "verified" only in a docstring -- had one thing in common: the
paper asserted a property that nothing checked. This recomputes every cell of
the §7.3 table from the jsonl files and fails on any disagreement, so a number
cannot drift from its data without the build noticing.

Growth-matched accuracy, its CI and the blind arm come from growth_matched.py's
subset definition; the numbers here are recomputed independently of whatever was
cached when the table was written.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict

import os as _os
R = _os.environ.get("MEDVIGIL3D_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
PAPER = R + "paper/PAPER.md"

TAG = {"SmolVLM2-2.2B": "smolvlm", "Qwen2.5-VL-3B": "qwen3b",
       "Qwen2.5-VL-7B": "qwen7b", "Qwen2.5-VL-32B": "qwen32b",
       "InternVL3-8B": "internvl", "InternVL3-14B": "internvl14",
       "Qwen3-VL-8B": "qwen3vl", "M3D-LaMed-Phi3-4B": "m3d",
       "M3D-LaMed-Llama2-7B": "m3dllama", "Med3DVLM-7B": "med3dvlm",
       "LLaVA-OneVision-7B": "llavaov", "Idefics3-8B": "idefics3",
       "Pixtral-12B": "pixtral", "Aria-25B-MoE": "aria"}


def arm_path(T: str, model: str, arm: str) -> str:
    """Prefer the explicitly-fill-labelled arm file over the un-suffixed one."""
    lab = f"{R}roi_{T}_{model}_{arm}_local.jsonl"
    return lab if os.path.exists(lab) else f"{R}roi_{T}_{model}_{arm}.jsonl"



def section(text: str, title: str, chapter: int) -> str:
    """Find a section by its title within a chapter, not by its full number.

    Section numbers shift whenever one is inserted -- adding the roadmap and the
    margin analysis moved every subsection of 7 by one, twice -- so a verifier
    keyed to "7.3" fails on a manuscript that is perfectly fine. Titles are
    stable. They are not unique, though: the 2D audit also has a "Headline
    results", and matching on the title alone silently pointed this checker at
    the wrong table. The chapter disambiguates without pinning the subsection.
    """
    hits = [m for m in re.finditer(
        rf"^#{{2,4}}\s+{chapter}\.\d+\s+{re.escape(title)}", text, re.M)]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} sections titled {title!r} in chapter "
                         f"{chapter}; expected exactly one")
    return text[hits[0].start():]


def vol(qid: str) -> str:
    return "_".join(qid.split("_")[:2])


def load(tag: str, arm: str):
    p = f"{R}mm_{tag}_{arm}.jsonl"
    if not os.path.exists(p):
        return None
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}


def matched_qids() -> set[str]:
    """The growth-matched subset, taken from the pipeline rather than restated.

    An earlier version of this checker reimplemented the balancing (exact growth
    values instead of 2 mm bins) and reported 25 disagreements that were entirely
    its own. A verifier that carries its own copy of a definition is a third
    source of truth, and disagreement with it says nothing about the paper.
    """
    sys.path.insert(0, R)
    from growth_matched import matched_subset, growth_of

    gold, growth = {}, growth_of()
    for l in open(R + "common_subset/qa/all.jsonl"):
        r = json.loads(l)
        gold[r["qid"]] = r["answer"]
    return matched_subset(gold, growth)


def acc(D, qids) -> float:
    qs = [q for q in qids if q in D]
    return 100 * sum(D[q]["prediction"] == D[q]["gold"] for q in qs) / len(qs)


def boot_ci(D, qids, n=2000, seed=0):
    byv = defaultdict(list)
    for q in qids:
        if q in D:
            byv[vol(q)].append(q)
    vols = sorted(byv)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [vols[rng.randrange(len(vols))] for _ in vols]
        qs = [q for v in s for q in byv[v]]
        out.append(100 * sum(D[q]["prediction"] == D[q]["gold"] for q in qs) / len(qs))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def pair_violation(D) -> float:
    p = defaultdict(list)
    for q, r in D.items():
        if r.get("pair_id"):
            p[r["pair_id"]].append(r)
    full = [v for v in p.values() if len(v) == 2]
    if not full:
        return float("nan")
    return 100 * sum(v[0]["prediction"] == v[1]["prediction"] for v in full) / len(full)



def verify_roi() -> int:
    """Recompute the per-organ VGR table of 7.4 from the arm files.

    The table churned more than any other during the audit -- a fill whose
    provenance could not be recovered, an arm short by 36 items, a second short
    by 154 -- so it is checked cell by cell like the headline table rather than
    trusted.
    """
    text = open(PAPER).read()
    blk = section(text, "Grounding: the ROI arms", 7)
    blk = blk[:blk.index("\n\n", blk.index("| Organ | Model |"))]
    TAG = {"M3D-Phi3": "m3d", "Qwen-7B": "qwen", "InternVL3-8B": "internvl",
           "Qwen-32B": "qwen32b"}
    ORGAN = {"Lung": "Task06_Lung", "Colon": "Task10_Colon",
             "Pancreas": "Task07_Pancreas", "Liver": "Task03_Liver"}
    bad = 0
    print(f"\n{'organ/model':28}{'arm':12}{'paper':>9}{'recomputed':>12}")
    for line in blk.splitlines():
        c = [x.replace("**", "").strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 5 or c[0] not in ORGAN or c[1] not in TAG:
            continue
        if "re-measur" in line or "running" in line:
            print(f"{c[0]+'/'+c[1]:28}{'(pending)':>33}")
            continue
        T, m = ORGAN[c[0]], TAG[c[1]]
        got = {}
        for arm, key in (("roi_only", "roi_only"), ("roi_masked", "roi_masked")):
            p_ = arm_path(T, m, key)
            if not os.path.exists(p_):
                got = None
                break
            D = {json.loads(l)["qid"]: json.loads(l) for l in open(p_)}
            got[arm] = D
        if got is None:
            print(f"{c[0]+'/'+c[1]:28}{'-- no data --':>33}")
            bad += 1
            continue
        keys = sorted(set(got["roi_only"]) & set(got["roi_masked"]))
        if set(got["roi_only"]) != set(got["roi_masked"]):
            print(f"{c[0]+'/'+c[1]:28}{'arms cover different items':>33}")
            bad += 1
            continue
        acc = {a: 100 * sum(D[q]["prediction"] == D[q]["gold"] for q in keys) / len(keys)
               for a, D in got.items()}
        for lbl, want, val in (("roi_only", c[2], acc["roi_only"]),
                               ("roi_masked", c[3], acc["roi_masked"]),
                               ("VGR", c[4], acc["roi_only"] - acc["roi_masked"]),
                               ("n", c[5], float(len(keys)))):
            w = float(want.replace("%", "").replace("+", "").replace("\u2212", "-").strip())
            ok = abs(val - w) <= 0.1
            if not ok:
                bad += 1
            print(f"{(c[0]+'/'+c[1]) if lbl=='roi_only' else '':28}{lbl:12}"
                  f"{w:9.1f}{val:12.1f}  {'' if ok else '<-- MISMATCH'}")
    return bad



def verify_fill_table() -> int:
    """Check the air-vs-local fill comparison against its files.

    These four numbers were computed by hand, at a moment when one of the four
    runs was still being written, and the partial value went into the paper.
    Every other table in this manuscript is recomputed by this script; this one
    was not, which is the only reason the error survived.
    """
    text = open(PAPER).read()
    blk = text[text.index("| InternVL3-8B | fill | roi_only |"):]
    blk = blk[:blk.index("\n\n")]
    ORG = {"Lung": ("Task06_Lung", 394), "Colon": ("Task10_Colon", 600),
           "Pancreas": ("Task07_Pancreas", 600),
           "Liver": ("Task03_Liver", 600)}
    bad = 0
    print(f"\n{'organ/fill':22}{'arm':12}{'paper':>9}{'recomputed':>12}")
    for line in blk.splitlines():
        m = re.match(r"\|\s*([A-Za-z]+)\s*\(n = [\d,]+\)\s*\|\s*\*{0,2}(air|local)[^|]*\|"
                     r"\s*([\d.]+)\s*%\s*\|\s*\*{0,2}([\d.]+)\s*%\*{0,2}\s*\|", line)
        if not m:
            continue
        organ, fill, only_p, masked_p = m.groups()
        T, n = ORG[organ]
        got = {}
        for arm in ("roi_only", "roi_masked"):
            path = f"{R}roi_{T}_internvl_{arm}_{fill}.jsonl"
            if not os.path.exists(path):
                print(f"{organ+'/'+fill:22}{'-- not run --':>33}")
                bad += 1
                got = None
                break
            D = {json.loads(l)["qid"]: json.loads(l) for l in open(path)}
            if len(D) != n:
                print(f"{organ+'/'+fill:22}{'incomplete: '+str(len(D))+'/'+str(n):>33}")
                bad += 1
                got = None
                break
            got[arm] = 100 * sum(v["prediction"] == v["gold"]
                                 for v in D.values()) / len(D)
        if got is None:
            continue
        for arm, want in (("roi_only", only_p), ("roi_masked", masked_p)):
            ok = abs(got[arm] - float(want)) <= 0.1
            if not ok:
                bad += 1
            print(f"{(organ+'/'+fill) if arm=='roi_only' else '':22}{arm:12}"
                  f"{float(want):9.1f}{got[arm]:12.1f}  {'' if ok else '<-- MISMATCH'}")
    return bad



def _verify_figure_csv(name: str, fields: tuple) -> int:
    """A table promoted to a figure still has to match its data.

    Moving a table into a figure removes it from this checker's reach, which
    would quietly retire the check along with the table. The figure's CSV is
    verified against the same summary instead.
    """
    import csv as _csv
    import json as _json
    path = R + "figdata/" + name
    src = R + "margin_summary.json"
    if not (os.path.exists(path) and os.path.exists(src)):
        print(f"  figure data {name} missing — cannot verify Figure 10")
        return 1
    got = _json.load(open(src))
    bad = 0
    for row in _csv.DictReader(open(path)):
        tag = TAG.get(row["model"])
        if tag not in got:
            continue
        for f in fields:
            if abs(float(row[f]) - got[tag][f]) > 0.06:
                print(f"  {name}: {row['model']} {f} "
                      f"{row[f]} vs {got[tag][f]:.3f} <-- MISMATCH")
                bad += 1
    print(f"  {name}: verified against margin_summary.json"
          if not bad else f"  {name}: {bad} mismatch(es)")
    return bad


def verify_margin() -> int:
    """Check the perturbation / gap / flip-rate table against margin_summary.json.

    Every row of this table was hand-transcribed. One of them was hand-*invented*:
    a Pixtral row reading 0.884 / 0.699 / 41.1 % was written into the manuscript
    before the analysis output was read, against actual values of 1.011 / 0.437 /
    57.3 %. Nothing but this function would have caught it.
    """
    import json as _json
    path = R + "margin_summary.json"
    if not os.path.exists(path):
        return 0
    got = _json.load(open(path))
    text = open(PAPER).read()
    if "| Model | Input | Perturbation |" not in text:
        # the table became Figure 10; its data file is still checked below
        return _verify_figure_csv("fig10_margin.csv",
                                  ("perturbation", "gap", "flip_rate"))
    blk = text[text.index("| Model | Input | Perturbation |"):]
    blk = blk[:blk.index("\n\n")]
    bad = 0
    print(f"\n{'model':22}{'quantity':14}{'paper':>9}{'recomputed':>12}")
    for line in blk.splitlines():
        c = [x.replace("**", "").strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 5 or c[0] not in TAG:
            continue
        tag = TAG[c[0]]
        if tag not in got:
            continue
        for label, want, val in (("perturbation", c[2], got[tag]["perturbation"]),
                                 ("decision gap", c[3], got[tag]["gap"]),
                                 ("flip rate", c[4].rstrip(" %"), got[tag]["flip_rate"])):
            w = float(want)
            ok = abs(val - w) <= 0.06
            if not ok:
                bad += 1
            print(f"{c[0] if label=='perturbation' else '':22}{label:14}"
                  f"{w:9.3f}{val:12.3f}  {'' if ok else '<-- MISMATCH'}")
    return bad


def main() -> None:
    text = open(PAPER).read()
    block = section(text, "Headline results", 7)
    block = block[:block.index("\n\n", block.index("| Model | Input |"))]

    matched = matched_qids()
    print(f"growth-matched subset: {len(matched)} items\n")
    print(f"{'model':22}{'column':16}{'paper':>9}{'recomputed':>12}  ")
    bad = 0
    for line in block.splitlines():
        m = re.match(r"\|\s*([A-Za-z0-9.\-+ ]+?)\s*\|\s*(montage|native)\s*\|(.+)\|", line)
        if not m:
            continue
        name = m[1].replace("**", "").strip()
        if name not in TAG:
            print(f"  !! unknown model row: {name}")
            bad += 1
            continue
        cells = [c.replace("**", "").replace("%", "").replace("−", "-").strip()
                 for c in m[3].split("|")]
        if "re-measur" in line or "*running*" in line:
            print(f"{name:22}{'(pending re-measurement)':>37}")
            continue
        S, B = load(TAG[name], "sighted"), load(TAG[name], "blind")
        if S is None or B is None:
            print(f"{name:22}{'-- no data on disk --':>37}")
            bad += 1
            continue
        full_p, matched_p, ci_p, blind_p, gain_p, viol_p = cells[:6]
        got = {
            "full corpus": acc(S, set(S)),
            "growth-matched": acc(S, matched),
            "matched blind": acc(B, matched),
            "image gain": acc(S, matched) - acc(B, matched),
            "pair viol.": pair_violation(S),
        }
        want = {"full corpus": float(full_p), "growth-matched": float(matched_p),
                "matched blind": float(blind_p), "image gain": float(gain_p),
                "pair viol.": float(viol_p)}
        lo, hi = boot_ci(S, matched)
        pl, ph = [float(x) for x in re.findall(r"-?\d+\.\d+", ci_p)]
        for k in want:
            ok = abs(got[k] - want[k]) <= 0.1
            if not ok:
                bad += 1
            print(f"{name if k=='full corpus' else '':22}{k:16}"
                  f"{want[k]:9.1f}{got[k]:12.1f}  {'' if ok else '<-- MISMATCH'}")
        ok = abs(lo - pl) <= 0.6 and abs(hi - ph) <= 0.6
        if not ok:
            bad += 1
        print(f"{'':22}{'95% CI':16}{f'[{pl},{ph}]':>9}{f'[{lo:.1f},{hi:.1f}]':>12}"
              f"  {'' if ok else '<-- MISMATCH'}")

    bad += verify_roi()
    bad += verify_fill_table()
    bad += verify_margin()
    print(f"\n{'HEADLINE AND ROI TABLES VERIFIED' if not bad else f'{bad} CELL(S) DISAGREE'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
