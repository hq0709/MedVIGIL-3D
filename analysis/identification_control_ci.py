"""
Scoring for the identification control (E1) and the solvability ceiling (E2).

Why this file exists
--------------------
EXPERIMENTS_TO_RUN.md says to score the control with
`analysis/image_contribution_ci.py` because "output format matches
`mm_*`/`roi_*` so it reads unchanged". The row format does match, but that
script cannot read these runs:

  * it enumerates files by the hardcoded pattern `mm_{tag}_sighted.jsonl` /
    `mm_{tag}_blind.jsonl`, and the control writes `id_{scope}_{tag}_{cond}.jsonl`;
  * it requires each file to hold the full 2,262-probe common subset and drops
    anything shorter, while the control is run per organ on the growth-matched
    subset (1,368 probes total, 274-383 per organ);
  * its contrast is sighted-minus-blind. The control's contrast is
    condition-minus-`plain`, and its acceptance criterion is whether a single
    condition's interval still contains 50 %.

So the statistics are reimplemented here, deliberately with the same design as
`image_contribution_ci.py`: probes are clustered in volumes, so the resampling
unit is the volume, the paired contrast is resampled inside the draw, and
B/seed match. Nothing here re-derives a subset: it reports exactly the items
present in the files it is given.

Usage
-----
    python analysis/identification_control_ci.py [--dir results_new] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

B = 10000
SEED = 0
CHANCE = 50.0                     # the yes/no probes


def chance_of(rows: list[dict]) -> float:
    """Chance rate for a run, read from the items rather than assumed.

    The counterfactual probes are two-way, but the E3 sub-tasks are four-way and
    scoring them against 50% would report every one of them as a catastrophic
    failure. Runs that carry their option list say what chance is; runs that do
    not are the two-way probes.
    """
    sizes = {len(r["choices"]) for r in rows if r.get("choices")}
    if len(sizes) == 1:
        return 100.0 / sizes.pop()
    return CHANCE

# id_{scope}_{model}_{condition}.jsonl -- scope is an MSD task name or "common",
# model is a run_multimodel tag (or "geometry" for the model-free oracle), and
# the condition is the only field that may itself contain a hyphen.
NAME = re.compile(r"^id_(?P<scope>[A-Za-z0-9]+(?:_[A-Za-z]+)?)_"
                  r"(?P<model>[a-z0-9]+)_(?P<cond>[a-z-]+)\.jsonl$")

ORDER = ["plain", "bestslice", "overlay", "identified",
         "text-oracle-blind", "text-oracle", "geometry-oracle"]


def volume_of(qid: str) -> str:
    return "_".join(qid.split("_")[:2])


def load(path: str) -> dict[str, dict]:
    return {r["qid"]: r for r in
            (json.loads(l) for l in open(path) if l.strip())}


def cluster_ci(by_vol: dict[str, list[float]], scale: float = 100.0):
    """Volume-level cluster bootstrap of the mean of per-probe values."""
    vols = sorted(by_vol)
    arrs = [np.asarray(by_vol[v], dtype=float) for v in vols]
    obs = scale * float(np.concatenate(arrs).mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vols), size=(B, len(vols)))
    boot = np.empty(B)
    for i in range(B):
        boot[i] = scale * float(np.concatenate([arrs[j] for j in idx[i]]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, float(lo), float(hi)


def complete_pairs(rows: list[dict]) -> list[list[dict]]:
    """Matched pairs with both members present in this run.

    The growth-matched subset is drawn per growth bin, not per pair, so a pair
    can lose one member to the matching: of 1,005 pairs in the common subset,
    363 survive complete and 642 items are left as singletons. That split
    matters, because the two halves are not equally hard.

    Within a pair, gap_mm is identical and only the growth amount differs, so
    the only way to get both members right is to compare the two numbers.
    Across singletons the gap is free to vary, and gap DOES correlate with the
    label -- "yes" means gap <= growth, so yes-items have smaller gaps -- and
    the matching removed the growth cue, not the gap cue. Any arm that is told
    the gap (all the text ceiling arms are) can therefore score above chance on
    singletons through gap magnitude alone, without ever comparing anything.
    Qwen2.5-VL-32B does exactly that: 51.1% on complete pairs, 73.7% on
    singletons, 61.7% pooled.

    So `acc_pairs` below is the endpoint to read for the ceiling arms, and the
    pooled figure is reported beside it rather than instead of it.
    """
    by_pair = defaultdict(list)
    for r in rows:
        if r.get("pair_id"):
            by_pair[r["pair_id"]].append(r)
    return [v for v in by_pair.values() if len(v) == 2]


def pair_stats(rows: list[dict]) -> tuple[float, float, int]:
    """(% of complete pairs answered identically, accuracy on them, n pairs)."""
    full = complete_pairs(rows)
    if not full:
        return float("nan"), float("nan"), 0
    same = sum(1 for v in full if v[0]["prediction"] == v[1]["prediction"])
    ok = sum(r["prediction"] == r["gold"] for v in full for r in v)
    return 100.0 * same / len(full), 100.0 * ok / (2 * len(full)), len(full)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dir", default=os.path.join(here, "results_new"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--scope", default=None,
                    help="restrict to one scope (e.g. Task03_Liver, common)")
    args = ap.parse_args()

    runs: dict[tuple[str, str, str], dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "id_*.jsonl"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            print(f"  ignoring unparseable name {os.path.basename(path)}")
            continue
        if args.scope and m["scope"] != args.scope:
            continue
        runs[(m["model"], m["cond"], m["scope"])] = load(path)
    if not runs:
        raise SystemExit(f"no id_*.jsonl runs under {args.dir}")

    # Per (model, condition), pool the organ-level files: the subset is
    # growth-matched corpus-wide, so the four organs together are the reported
    # unit and each organ alone is a slice of it.
    pooled: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    scopes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (model, cond, scope), rows in runs.items():
        pooled[(model, cond)].update(rows)
        scopes[(model, cond)].add(scope)

    def rank(key):
        model, cond = key
        return (model, ORDER.index(cond) if cond in ORDER else len(ORDER), cond)

    out_rows = []
    print("Accuracy per condition, volume-level cluster bootstrap "
          f"(B={B:,}, seed={SEED})\n")
    print(f"{'model':10}{'condition':19}{'n':>6}{'vols':>5}{'acc':>7}"
          f"{'95% CI':>16}{'>chance':>8}   {'pairs':>6}{'acc|pairs':>10}"
          f"{'95% CI':>16}{'>chance':>8}{'same':>7}")
    print("-" * 116)
    for key in sorted(pooled, key=rank):
        model, cond = key
        rows = list(pooled[key].values())
        by_vol = defaultdict(list)
        for r in rows:
            by_vol[volume_of(r["qid"])].append(float(r["prediction"] == r["gold"]))
        acc, lo, hi = cluster_ci(by_vol)
        chance = chance_of(rows)
        excl = "yes" if lo > chance else ("below" if hi < chance else "no")
        same, acc_p, n_pairs = pair_stats(rows)

        # the pair-restricted endpoint gets its own interval, clustered the same
        # way, so the two columns are comparable
        pv = defaultdict(list)
        for pair in complete_pairs(rows):
            for r in pair:
                pv[volume_of(r["qid"])].append(float(r["prediction"] == r["gold"]))
        if pv:
            _, plo, phi = cluster_ci(pv)
            pexcl = "yes" if plo > chance else ("below" if phi < chance else "no")
            pair_cols = (f"{n_pairs:>6}{acc_p:>9.1f}%  [{plo:5.1f},{phi:5.1f}]"
                         f"{pexcl:>8}{same:>6.0f}%")
        else:
            plo = phi = float("nan")
            pexcl = "n/a"
            pair_cols = f"{0:>6}{'—':>10}{'—':>16}{'—':>8}{'—':>7}"
        print(f"{model:10}{cond:19}{len(rows):>6}{len(by_vol):>5}{acc:>6.1f}%"
              f"  [{lo:5.1f},{hi:5.1f}]{excl:>8}   {pair_cols}")
        out_rows.append({"model": model, "condition": cond,
                         "scopes": "+".join(sorted(scopes[key])),
                         "n_probes": len(rows), "n_volumes": len(by_vol),
                         "accuracy": round(acc, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "chance": chance,
                         "excludes_chance": excl,
                         "n_complete_pairs": n_pairs,
                         "accuracy_pairs_only": round(acc_p, 2),
                         "pairs_ci_lo": round(plo, 2), "pairs_ci_hi": round(phi, 2),
                         "pairs_excludes_chance": pexcl,
                         "pair_identical_pct": round(same, 2),
                         "delta_vs_plain": "", "delta_ci_lo": "",
                         "delta_ci_hi": "", "delta_excludes_zero": ""})

    # Paired contrast against `plain`, the reproduction of the published input.
    print("\nPaired difference from `plain` on the probes both arms scored")
    print(f"{'model':10}{'condition':19}{'n':>7}{'vols':>6}{'delta':>9}"
          f"{'95% CI':>18}{'excl. 0':>9}")
    print("-" * 78)
    idx = {r["model"] + "|" + r["condition"]: r for r in out_rows}
    for key in sorted(pooled, key=rank):
        model, cond = key
        if cond == "plain" or (model, "plain") not in pooled:
            continue
        A, P = pooled[key], pooled[(model, "plain")]
        qs = sorted(set(A) & set(P))
        if not qs:
            continue
        by_vol = defaultdict(list)
        for q in qs:
            by_vol[volume_of(q)].append(
                float(A[q]["prediction"] == A[q]["gold"])
                - float(P[q]["prediction"] == P[q]["gold"]))
        d, lo, hi = cluster_ci(by_vol)
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"{model:10}{cond:19}{len(qs):>7}{len(by_vol):>6}{d:>+8.1f}"
              f"  [{lo:+5.1f},{hi:+5.1f}]{excl:>9}")
        row = idx[model + "|" + cond]
        row.update({"delta_vs_plain": round(d, 2), "delta_ci_lo": round(lo, 2),
                    "delta_ci_hi": round(hi, 2), "delta_excludes_zero": excl})

    dest = args.csv or os.path.join(args.dir, "identification_control_ci.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {dest}")
    print("`>chance` reads yes / no / below: whether the volume-clustered 95% "
          "interval lies entirely above the run's chance rate (50% for the "
          "yes/no probes, 25% for the four-way sub-tasks), contains it, or "
          "lies below it.")
    print("`acc|pairs` is accuracy restricted to matched pairs with both members "
          "present -- the only items where gap_mm is held fixed, so the only "
          "ones an answer cannot reach through gap magnitude. `same` is the "
          "share of those pairs answered identically, which geometry forbids.")


if __name__ == "__main__":
    main()
