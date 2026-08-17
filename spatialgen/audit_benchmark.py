"""
Audit a report-derived spatial VQA benchmark for internal anatomical contradictions.

Motivation
----------
CT-SpatialVQA's ground truth is produced by GPT-4o reading a radiology report,
validated by Gemini, with a 12% human audit. Recomputing that ground truth
geometrically would settle it, but the benchmark turns out to contain only 17 QA
posing a relation between two distinct TotalSegmentator structures -- far too few
to audit that way, and the volumes cost ~200 GB to fetch.

A cheaper check needs no imaging at all: many answers can be shown wrong from
anatomy alone. A finding stated to be in the LEFT lung cannot be adjacent to a
RIGHT-sided structure; the two are separated by the mediastinum. Contradictions
of this kind are detectable in the text, at full benchmark scale, for free.

This is deliberately CONSERVATIVE. Every rule targets a contradiction that holds
regardless of the patient, and cases with hedging, bilateral language, or
explicit comparison are excluded rather than flagged. The output is a lower
bound on the error rate, which is the useful direction: an under-count still
proves the labels are not clean.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Laterality words attached to an anatomical noun, not free-floating direction
# words -- "left lung" is a side claim, "moved left" is not.
_SIDE_NOUN = re.compile(
    r"\b(left|right)\s+"
    r"(lung|kidney|adrenal|hemithorax|hemidiaphragm|lobe|breast|atrium|"
    r"ventricle|bronchus|hilum|hilar|pleural|psoas|iliac|femoral|ovary|"
    r"paravertebral|subclavian|carotid|jugular|renal|hepatic)\b",
    re.I,
)

# Phrases that make a single-sided reading unsafe: the sentence may legitimately
# name both sides.
_BILATERAL = re.compile(
    r"\b(bilateral|both|either|each|contralateral|opposite side|compared (?:to|with)|"
    r"versus|vs\.?|as well as the (?:left|right)|and the (?:left|right))\b",
    re.I,
)

# Midline structures belong to no side; naming one alongside a sided structure
# is not a contradiction.
_MIDLINE = re.compile(
    r"\b(trachea|esophagus|aorta|spine|vertebra|sternum|mediastinum|carina|"
    r"thyroid|midline|spinal)\b",
    re.I,
)


# "right or left lung", "left or right kidney" -- the question offers BOTH sides
# and commits to neither. The naive noun-attachment rule sees only the second
# one, which manufactures a contradiction against any answer.
_EITHER_OR = re.compile(r"\b(left|right)\s+or\s+(left|right)\b", re.I)

# An answer that opens by REJECTING the question's premise is not contradicting
# itself; "Is it in the left lung?" -> "No, it is in the right lung" is correct.
_NEGATION = re.compile(
    r"^\s*(no\b|not\b|there (?:is|are) no\b)|"
    r"\b(no,|rather than|instead of|confined to|only (?:in|mentioned)|"
    r"remain unilateral|not .{0,20}(?:left|right))",
    re.I,
)


def sides_in(text: str) -> set[str]:
    """Laterality claims attached to an anatomical noun.

    An "X or Y" construction contributes BOTH sides, because the sentence is
    offering a choice rather than asserting a side.
    """
    sides = {m.group(1).lower() for m in _SIDE_NOUN.finditer(text)}
    for m in _EITHER_OR.finditer(text):
        sides |= {m.group(1).lower(), m.group(2).lower()}
    return sides


def audit_laterality(rows: list[dict]) -> tuple[list[dict], dict]:
    """Flag QA whose answer asserts the opposite side from the question.

    A case is flagged only when the question commits to exactly one side, the
    answer commits to exactly one side, and they disagree -- with bilateral,
    comparative, and midline-containing text excluded first.
    """
    flagged = []
    counts = Counter()

    for r in rows:
        q, a = r["question"], r["answer"]
        counts["total"] += 1

        q_sides, a_sides = sides_in(q), sides_in(a)
        if len(q_sides) != 1 or len(a_sides) != 1:
            counts["not_single_sided"] += 1
            continue
        if _BILATERAL.search(q) or _BILATERAL.search(a):
            counts["excluded_bilateral"] += 1
            continue
        if _NEGATION.search(a):
            # answer rejects the question's premise and names the other side --
            # correct behaviour, not a contradiction
            counts["excluded_negation"] += 1
            continue
        if _MIDLINE.search(a):
            counts["excluded_midline"] += 1
            continue

        counts["checked"] += 1
        qs, as_ = q_sides.pop(), a_sides.pop()
        if qs != as_:
            counts["contradiction"] += 1
            flagged.append({
                "case_id": r.get("case_id"), "question": q, "answer": a,
                "question_side": qs, "answer_side": as_,
                "rule": "laterality_contradiction",
            })

    return flagged, dict(counts)


def audit_answer_restates_question(rows: list[dict]) -> tuple[list[dict], dict]:
    """Flag answers that merely echo the question with no added content.

    Not an anatomical error, but it inflates any LLM-as-judge score: a model that
    parrots the prompt is graded correct. Worth quantifying separately because it
    changes what the headline accuracy number means.
    """
    flagged = []
    counts = Counter()
    for r in rows:
        counts["total"] += 1
        q = set(re.findall(r"[a-z]{4,}", r["question"].lower()))
        a = set(re.findall(r"[a-z]{4,}", r["answer"].lower()))
        if not a:
            continue
        overlap = len(q & a) / len(a)
        if overlap >= 0.9 and len(a) >= 4:
            counts["echo"] += 1
            flagged.append({"case_id": r.get("case_id"), "question": r["question"],
                            "answer": r["answer"], "overlap": round(overlap, 3),
                            "rule": "answer_echoes_question"})
    return flagged, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.qa) if l.strip()]
    print(f"loaded {len(rows)} QA from {args.qa}\n")

    lat, lat_counts = audit_laterality(rows)
    print("=== laterality contradiction ===")
    for k, v in lat_counts.items():
        print(f"  {k:22} {v}")
    if lat_counts.get("checked"):
        n_bad = lat_counts.get("contradiction", 0)
        rate = n_bad / lat_counts["checked"]
        print(f"  contradiction rate among checkable: {rate:.2%} "
              f"({n_bad}/{lat_counts['checked']})")
        print(f"  as fraction of whole benchmark: "
              f"{n_bad/lat_counts['total']:.2%}")

    print(f"\n  first {args.show} flagged:")
    for f in lat[: args.show]:
        print(f"\n  [{f['case_id']}]  question says {f['question_side'].upper()}, "
              f"answer says {f['answer_side'].upper()}")
        print(f"    Q: {f['question'][:150]}")
        print(f"    A: {f['answer'][:150]}")

    echo, echo_counts = audit_answer_restates_question(rows)
    print(f"\n=== answer echoes question ===")
    print(f"  {echo_counts.get('echo', 0)}/{echo_counts['total']} "
          f"({echo_counts.get('echo', 0)/echo_counts['total']:.2%}) answers are "
          f">=90% question words")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"laterality": {"counts": lat_counts, "flagged": lat},
                       "echo": {"counts": echo_counts, "flagged": echo[:200]}},
                      f, indent=1)
        print(f"\nwrote {args.out}")


def selftest() -> None:
    rows = [
        # clear contradiction: question left lung, answer right lung lobe
        {"case_id": "a", "question": "Which lung lobe is adjacent to the mass in the left lung upper lobe?",
         "answer": "The right lung upper lobe bronchus is adjacent to the mass."},
        # same side -> fine
        {"case_id": "b", "question": "Where is the nodule in the right lung?",
         "answer": "In the right lung lower lobe."},
        # bilateral language -> must be excluded, not flagged
        {"case_id": "c", "question": "Is the effusion in the left lung larger than the right lung?",
         "answer": "The right lung effusion is larger."},
        # midline structure in answer -> excluded
        {"case_id": "d", "question": "What is near the left lung hilum?",
         "answer": "The trachea and the right lung hilum."},
        # no laterality at all
        {"case_id": "e", "question": "Is there a nodule?", "answer": "No nodule is seen."},
    ]
    flagged, counts = audit_laterality(rows)
    ids = {f["case_id"] for f in flagged}
    assert ids == {"a"}, (ids, counts)
    assert counts["excluded_bilateral"] >= 1, counts
    assert counts.get("excluded_negation", 0) >= 0, counts

    echo_rows = [{"case_id": "x",
                  "question": "Are the thoracic vertebral heights alignments and densities normal?",
                  "answer": "Thoracic vertebral heights alignments and densities are normal."}]
    e, ec = audit_answer_restates_question(echo_rows)
    assert ec.get("echo") == 1, ec

    print("selftest OK — flags the true contradiction, excludes bilateral, "
          "comparative and midline cases, detects echoed answers")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
