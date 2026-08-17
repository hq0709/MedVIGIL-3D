"""
Aggregate pipeline output into the numbers that go in the writeup.

Reports the things a reviewer will ask about first, including the ones that are
unflattering: how often the anatomy segmentation failed, how often a binding was
refused as ambiguous, and whether every answer still reproduces from its
provenance. Silent truncation of those is how a dataset paper becomes wrong.

  python summarize.py --outdir ../out_lung [--outdir ../out_spleen]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qa_gen import validate_provenance  # noqa: E402


def load_jsonl_dir(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.jsonl")):
        with open(f) as fh:
            out.extend(json.loads(l) for l in fh if l.strip())
    return out


def summarize(outdir: Path) -> dict:
    qa_dir = outdir / "qa"
    qas = load_jsonl_dir(qa_dir) if qa_dir.is_dir() else []

    summary_path = outdir / "summary.json"
    run = json.load(open(summary_path)) if summary_path.exists() else {}
    stats, failures = run.get("stats", []), run.get("failures", [])

    rep = {
        "outdir": str(outdir),
        "volumes_ok": len(stats),
        "volumes_failed": len(failures),
        "qa_total": len(qas),
        "qa_by_category": dict(Counter(q.get("category") for q in qas)),
        "provenance_mismatches": len(validate_provenance(qas)),
    }

    if stats and "n_lesions" in stats[0]:          # lesion-binding run
        n_les = sum(s["n_lesions"] for s in stats)
        n_amb = sum(s["n_ambiguous"] for s in stats)
        rep["lesions_total"] = n_les
        rep["lesions_ambiguous"] = n_amb
        rep["ambiguity_rate"] = round(n_amb / n_les, 4) if n_les else None
        rep["container_distribution"] = dict(
            Counter(c for s in stats for c in s["containers"]))
        rep["volumes_with_no_lesion"] = sum(1 for s in stats if s["n_lesions"] == 0)
    else:                                          # anatomy scene-graph run
        if stats:
            rep["mean_structures"] = round(
                sum(s["n_structures"] for s in stats) / len(stats), 1)
            rep["mean_relations"] = round(
                sum(s["n_relations"] for s in stats) / len(stats), 1)
            rep["axiom_violations_own_graph"] = sum(
                s.get("axiom_violations", 0) for s in stats)

    if stats:
        rep["mean_seconds_per_volume"] = round(
            sum(s.get("seconds", s.get("total_seconds", 0)) for s in stats)
            / len(stats), 1)

    # inverse-pair coverage: how much of the antisymmetry probe actually exists
    inv = sum(1 for q in qas if q["qid"].endswith("_inv"))
    rep["inverse_pairs"] = inv

    rep["failure_examples"] = [f.get("error", "")[:120] for f in failures[:3]]
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", action="append", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    reports = [summarize(Path(d)) for d in args.outdir]
    for r in reports:
        print(f"\n=== {r['outdir']} ===")
        print(f"volumes: {r['volumes_ok']} ok / {r['volumes_failed']} failed"
              + (f"  ({r['volumes_failed']/(r['volumes_ok']+r['volumes_failed']):.1%} failure rate)"
                 if r["volumes_ok"] + r["volumes_failed"] else ""))
        print(f"QA: {r['qa_total']}  by category: {r['qa_by_category']}")
        print(f"inverse pairs (antisymmetry probe): {r['inverse_pairs']}")
        pm = r["provenance_mismatches"]
        print(f"provenance mismatches: {pm}" + ("  <-- MUST BE 0" if pm else "  (clean)"))
        if "lesions_total" in r:
            print(f"lesions: {r['lesions_total']}, ambiguous {r['lesions_ambiguous']} "
                  f"({r['ambiguity_rate']:.1%})" if r["lesions_total"] else "lesions: 0")
            print(f"volumes with no lesion found: {r['volumes_with_no_lesion']}")
            print(f"containers: {r['container_distribution']}")
        else:
            if "mean_structures" in r:
                print(f"mean structures/volume: {r['mean_structures']}, "
                      f"mean relations: {r['mean_relations']}")
                print(f"axiom violations in generated graphs: "
                      f"{r['axiom_violations_own_graph']} (must be 0)")
        if "mean_seconds_per_volume" in r:
            print(f"mean seconds/volume: {r['mean_seconds_per_volume']}")
        for e in r["failure_examples"]:
            print(f"  failure: {e}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
