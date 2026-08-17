"""
Corpus-level filter: remove relations that never vary across the dataset.

Why a second filter is needed
-----------------------------
The name-based filter in qa_gen catches relations decided by the words
("kidney_left is left of kidney_right"). It cannot catch relations decided by
canonical human anatomy: the liver is superior to the hip in every patient
alive, so "is the liver superior or inferior to the hip" needs no image either.
Measured after name-filtering, a blind text-only LLM still scored 70.0% on
longitudinal and 71.9% on laterality against a 50% chance level.

The general test does not require knowing any anatomy: if a (subject, object,
axis) triple resolves to the SAME predicate in every volume where both
structures appear, then that relation carries zero information about any
particular volume, whatever the underlying reason. Those triples are dropped.

Needs enough volumes to distinguish "constant" from "we only saw it once",
hence --min-volumes.

  python corpus_filter.py --outdir ../out_spleen --min-volumes 5 \
                          --write ../out_spleen/qa_filtered
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qa_gen import constant_relation_keys  # noqa: E402


def load_graphs(graphs_dir: Path, drop_truncated: bool = False
                ) -> tuple[list[list[dict]], int]:
    """Load per-volume relation lists.

    With drop_truncated, relations involving a structure whose segmentation
    touches the volume boundary are removed. Such a structure's centroid and
    bounding box describe only the captured fragment, so any relation computed
    from it reflects where the scan was cropped, not anatomy. In 12 chest CTs
    this single effect accounted for the ONE directional relation that appeared
    to vary across patients: vertebrae_C7 is truncated in every chest scan.
    """
    out, n_dropped = [], 0
    for f in sorted(graphs_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        rels = d.get("relations", [])
        if drop_truncated:
            trunc = {n for n, s in (d.get("structures") or {}).items()
                     if s.get("truncated")}
            if trunc:
                before = len(rels)
                rels = [r for r in rels
                        if r["subject"] not in trunc and r["object"] not in trunc]
                n_dropped += before - len(rels)
        out.append(rels)
    return out, n_dropped


def relation_key(qa: dict) -> tuple[str, str, str] | None:
    rel = (qa.get("provenance") or {}).get("relation")
    if not rel or not rel.get("axis"):
        return None
    return (rel["subject"], rel["object"], rel["axis"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-volumes", type=int, default=5)
    ap.add_argument("--write", default=None, help="dir to write the filtered QA")
    ap.add_argument("--drop-truncated", action="store_true",
                    help="exclude structures cut off by the field of view")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    per_vol, n_trunc_dropped = load_graphs(outdir / "graphs", args.drop_truncated)
    if args.drop_truncated:
        print(f"excluded {n_trunc_dropped} relations involving "
              f"field-of-view-truncated structures\n")
    if len(per_vol) < args.min_volumes:
        sys.exit(f"only {len(per_vol)} volumes with graphs; need >= "
                 f"{args.min_volumes} for the constancy test to mean anything")

    const = constant_relation_keys(per_vol, min_volumes=args.min_volumes)

    # Directional (axis) and adjacency relations are judged by different rules,
    # so they need separate denominators -- mixing them produced a "214.6% of
    # judgeable" figure, which is a sign the counts were incommensurable.
    counts: Counter = Counter()
    seen_pred = defaultdict(set)
    pair_seen: Counter = Counter()
    pair_adj: Counter = Counter()

    for rels in per_vol:
        structures = {r["subject"] for r in rels} | {r["object"] for r in rels}
        adj_here = set()
        for r in rels:
            if r.get("axis"):
                k = (r["subject"], r["object"], r["axis"])
                counts[k] += 1
                seen_pred[k].add(r["predicate"])
            elif r["predicate"] == "adjacent_to":
                adj_here.add(tuple(sorted((r["subject"], r["object"]))))
        for a in sorted(structures):
            for b in sorted(structures):
                if a < b:
                    pair_seen[(a, b)] += 1
                    if (a, b) in adj_here:
                        pair_adj[(a, b)] += 1

    ax_judgeable = {k for k, c in counts.items() if c >= args.min_volumes}
    ax_const = {k for k in ax_judgeable if len(seen_pred[k]) == 1}
    ax_vary = ax_judgeable - ax_const

    adj_judgeable = {p for p, n in pair_seen.items() if n >= args.min_volumes}
    adj_const = {p for p in adj_judgeable
                 if pair_adj.get(p, 0) in (0, pair_seen[p])}
    adj_vary = adj_judgeable - adj_const

    print(f"volumes: {len(per_vol)}\n")
    print(f"DIRECTIONAL relations (left/right, sup/inf, ant/post):")
    print(f"  judgeable (seen in >= {args.min_volumes} volumes): {len(ax_judgeable)}")
    print(f"  constant across patients (DROP): {len(ax_const)} "
          f"({len(ax_const)/max(len(ax_judgeable),1):.1%})")
    print(f"  genuinely varying (KEEP):        {len(ax_vary)} "
          f"({len(ax_vary)/max(len(ax_judgeable),1):.1%})")

    # A run made with --no-adjacency contains no adjacency edges at all. Every
    # pair then looks "never adjacent in every volume" and is scored 100%
    # constant -- a completely spurious result that reads exactly like a real
    # finding. Refuse to report the adjacency block rather than print it.
    total_adj_edges = sum(pair_adj.values())
    print(f"\nADJACENCY relations:")
    if total_adj_edges == 0:
        print("  NOT COMPUTED — these graphs contain zero adjacency edges.")
        print("  (the pipeline was run with --no-adjacency). Any 'constant'")
        print("  figure here would be an artefact of absence, not a measurement.")
    else:
        print(f"  structure pairs judgeable: {len(adj_judgeable)}")
        print(f"  always-or-never adjacent (DROP): {len(adj_const)} "
              f"({len(adj_const)/max(len(adj_judgeable),1):.1%})")
        print(f"  adjacency VARIES by patient (KEEP): {len(adj_vary)} "
              f"({len(adj_vary)/max(len(adj_judgeable),1):.1%})")

    print("\n examples of CONSTANT directional relations (no volumetric info):")
    for k in sorted(ax_const)[:6]:
        print(f"   {k[0]} -- {k[2]} --> {k[1]}   always {list(seen_pred[k])[0]}")
    print("\n examples of VARYING adjacency (patient-specific, the useful ones):")
    for p in sorted(adj_vary)[:8]:
        print(f"   {p[0]} <-> {p[1]}   adjacent in {pair_adj.get(p,0)}/{pair_seen[p]} volumes")

    # apply to the QA shards
    qa_dir = outdir / "qa"
    total = kept_n = 0
    kept_rows = defaultdict(list)
    dropped_by_cat: Counter = Counter()
    for f in sorted(qa_dir.glob("*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            qa = json.loads(line)
            total += 1
            k = relation_key(qa)
            if k is not None and k in const:
                dropped_by_cat[qa.get("category", "?")] += 1
                continue
            kept_n += 1
            kept_rows[f.name].append(qa)

    print(f"\nQA: {total} -> {kept_n} kept "
          f"({(total-kept_n)/max(total,1):.1%} dropped as constant)")
    if dropped_by_cat:
        print(f"  dropped by category: {dict(dropped_by_cat)}")

    if args.write:
        wd = Path(args.write)
        wd.mkdir(parents=True, exist_ok=True)
        for name, rows in kept_rows.items():
            with open(wd / name, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        print(f"wrote filtered QA to {wd}")


def selftest() -> None:
    # liver->hip is superior in every volume (constant); nodule->rib flips (varying)
    vols = []
    for i in range(6):
        vols.append([
            {"subject": "liver", "object": "hip_left", "axis": "longitudinal",
             "predicate": "superior_to"},
            {"subject": "massA", "object": "rib_left_3", "axis": "lateral",
             "predicate": "left_of" if i % 2 else "right_of"},
        ])
    const = constant_relation_keys(vols, min_volumes=5)
    assert ("liver", "hip_left", "longitudinal") in const, const
    assert ("massA", "rib_left_3", "lateral") not in const, const

    # a triple seen only twice must NOT be called constant on that evidence
    thin = [[{"subject": "a", "object": "b", "axis": "lateral",
              "predicate": "left_of"}] for _ in range(2)]
    assert not constant_relation_keys(thin, min_volumes=5)

    print("selftest OK — constant relations detected, varying ones kept, "
          "and thin evidence is not mistaken for constancy")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
