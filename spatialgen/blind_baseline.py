"""
Blind baseline: answer spatial questions with a TEXT-ONLY model and no image.

The point
---------
CT-SpatialVQA concludes that 3D medical VLMs "rely primarily on learned priors
and language correlations" rather than volumetric evidence. That diagnosis is
about the MODELS. But it can equally be a property of the QUESTIONS: if a
language model that never sees the scan answers well above chance, then the
questions are answerable from priors, and a VLM scoring above chance proves
nothing about grounding.

This is the standard blind-baseline control for a VQA benchmark, and it is cheap
-- text inference only. Run on two sets:

  * CT-SpatialVQA (report-derived) -- how much of it is answerable blind?
  * our geometry-derived QA        -- should sit AT chance, because the answer
                                      depends on this specific volume's anatomy

A generator that produces questions a blind model can guess is not producing
spatial supervision, so this doubles as a validity check on our own pipeline.

Scoring is restricted to items with a categorical gold answer (left/right,
yes/no) so that chance level is exactly defined and no LLM judge is needed --
a judge would introduce the very language-prior effect being measured.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_SIDE = re.compile(r"\b(left|right)\b", re.I)
_YESNO = re.compile(r"^\s*(yes|no)\b", re.I)


def categorical_gold(question: str, answer: str) -> tuple[str, list[str]] | None:
    """Reduce a free-text QA to (gold_label, choices), or None if not categorical.

    Only two families are admitted, both with an unambiguous 50% chance level:
      laterality  -- the question asks which side, the answer names exactly one
      yes/no      -- the answer opens with yes or no
    """
    q_sides = {m.group(1).lower() for m in _SIDE.finditer(question)}
    a_sides = {m.group(1).lower() for m in _SIDE.finditer(answer)}

    # laterality: question offers both sides (or asks "which side"), answer picks one
    asks_side = ("which side" in question.lower()
                 or "what side" in question.lower()
                 or len(q_sides) == 2)
    if asks_side and len(a_sides) == 1:
        return a_sides.pop(), ["left", "right"]

    m = _YESNO.match(answer)
    if m:
        return m.group(1).lower(), ["no", "yes"]
    return None


def load_ctspatialvqa(path: Path, limit: int, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        g = categorical_gold(r["question"], r["answer"])
        if g:
            out.append({"qid": f"{r['case_id']}::{len(out)}",
                        "question": r["question"], "gold": g[0],
                        "choices": g[1], "source": "ct_spatialvqa"})
    random.Random(seed).shuffle(out)
    return out[:limit]


def load_ours(qa_dir: Path, limit: int, seed: int = 0) -> list[dict]:
    out = []
    for f in sorted(qa_dir.glob("*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            ch = r.get("choices")
            if ch and len(ch) == 2 and r["answer"] in ch:
                out.append({"qid": r["qid"], "question": r["question"],
                            "gold": r["answer"], "choices": ch,
                            "category": r.get("category"), "source": "ours"})
    random.Random(seed).shuffle(out)
    return out[:limit]


def ask(endpoint: str, model: str, question: str, choices: list[str],
        timeout: int = 60) -> str:
    prompt = (
        "You are shown NO image. Answer the following question about a chest CT "
        "using your best judgement.\n"
        f"Question: {question}\n"
        f"Answer with exactly one of: {', '.join(choices)}. "
        "Reply with the single word only."
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"]


def normalise(text: str, choices: list[str]) -> str | None:
    low = (text or "").lower()
    for c in choices:
        if re.search(rf"\b{re.escape(c)}\b", low):
            return c
    return None


def run(items: list[dict], endpoint: str, model: str,
        record_path: str | None = None) -> dict:
    correct = parsed = 0
    per_cat = Counter()
    per_cat_n = Counter()
    pred_dist = Counter()
    errors = 0
    records = []

    for i, it in enumerate(items, 1):
        try:
            raw = ask(endpoint, model, it["question"], it["choices"])
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  request error: {e}", file=sys.stderr)
            continue
        p = normalise(raw, it["choices"])
        records.append({"qid": it["qid"], "gold": it["gold"], "pred": p,
                        "raw": raw, "category": it.get("category")})
        if p is None:
            continue
        parsed += 1
        pred_dist[p] += 1
        cat = it.get("category", "all")
        per_cat_n[cat] += 1
        if p == it["gold"]:
            correct += 1
            per_cat[cat] += 1
        if i % 50 == 0:
            print(f"    {i}/{len(items)} ...", flush=True)

    if record_path:
        with open(record_path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    return {
        "n": len(items), "parsed": parsed, "correct": correct, "errors": errors,
        "accuracy": correct / parsed if parsed else None,
        "prediction_distribution": dict(pred_dist),
        "gold_distribution": dict(Counter(it["gold"] for it in items)),
        "per_category": {c: {"n": per_cat_n[c], "correct": per_cat[c],
                             "accuracy": per_cat[c] / per_cat_n[c]}
                         for c in per_cat_n},
    }


def trivial_baseline(items: list[dict]) -> dict:
    """Best score obtainable without reading any question content.

    A single global majority label is WRONG here: different items offer
    different choice pairs (left/right, superior/inferior, yes/no), so the
    global mode of a 7-label pool understates chance badly -- it reported 15.2%
    for a set where every item is binary and chance is 50%.

    The right reference is the best of:
      * random guessing among each item's own choices, averaged
      * always answering the majority label WITHIN each choice-pair group
    """
    by_pair: dict[tuple, Counter] = {}
    for it in items:
        key = tuple(sorted(it["choices"]))
        by_pair.setdefault(key, Counter())[it["gold"]] += 1

    n = len(items)
    random_rate = sum(1.0 / len(it["choices"]) for it in items) / n if n else 0.0
    majority_rate = sum(max(c.values()) for c in by_pair.values()) / n if n else 0.0
    return {"random": random_rate, "per_pair_majority": majority_rate,
            "reference": max(random_rate, majority_rate),
            "choice_pairs": {"/".join(k): sum(v.values()) for k, v in by_pair.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--ctspatialvqa", default=None)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--choice-pair", default=None,
                    help="restrict to one choice pair, e.g. left/right")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for name, loader, src in (
        ("ct_spatialvqa", load_ctspatialvqa, args.ctspatialvqa),
        ("ours", load_ours, args.ours),
    ):
        if not src:
            continue
        items = loader(Path(src), 10**9)
        if args.choice_pair:
            want = tuple(sorted(args.choice_pair.split("/")))
            items = [i for i in items if tuple(sorted(i["choices"])) == want]
        items = items[: args.limit]
        if not items:
            print(f"{name}: no categorical items found")
            continue
        tb = trivial_baseline(items)
        mc = tb["reference"]
        print(f"\n=== {name}: {len(items)} categorical items ===", flush=True)
        print(f"  trivial baselines: random {tb['random']:.1%}, "
              f"per-pair majority {tb['per_pair_majority']:.1%} "
              f"-> reference {mc:.1%}", flush=True)
        print(f"  choice pairs: {tb['choice_pairs']}", flush=True)
        r = run(items, args.endpoint, args.model,
                record_path=(f"{args.out}.{name}.records.jsonl" if args.out else None))
        r["trivial_baseline"] = tb
        results[name] = r
        acc = r["accuracy"]
        print(f"  blind accuracy: {acc:.1%} ({r['correct']}/{r['parsed']})"
              if acc is not None else "  no parseable predictions")
        if acc is not None:
            print(f"  vs majority-class {mc:.1%}  -> "
                  f"{'ABOVE (answerable from priors)' if acc > mc + 0.05 else 'at/below chance'}")
        print(f"  gold dist: {r['gold_distribution']}")
        print(f"  pred dist: {r['prediction_distribution']}")
        if r["per_category"]:
            for c, v in sorted(r["per_category"].items()):
                print(f"    {c:18} {v['accuracy']:6.1%}  n={v['n']}")

    if args.out and results:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")


def selftest() -> None:
    assert categorical_gold("Is the nodule in the right or left lung?",
                            "In the right lung.") == ("right", ["left", "right"])
    assert categorical_gold("On which side is the effusion?",
                            "The effusion is on the left.") == ("left", ["left", "right"])
    assert categorical_gold("Is there a nodule?", "Yes, in the apex.")[0] == "yes"
    # not categorical: no side asked, no yes/no
    assert categorical_gold("Where is the trachea?", "In the midline.") is None
    # ambiguous: answer names both sides
    assert categorical_gold("Is it in the right or left lung?",
                            "In both the right and left lungs.") is None

    assert normalise("The answer is LEFT.", ["left", "right"]) == "left"
    assert normalise("banana", ["left", "right"]) is None

    items = ([{"gold": "yes", "choices": ["no", "yes"]}] * 8
             + [{"gold": "no", "choices": ["no", "yes"]}] * 2)
    tb = trivial_baseline(items)
    assert abs(tb["random"] - 0.5) < 1e-9, tb
    assert abs(tb["per_pair_majority"] - 0.8) < 1e-9, tb
    # mixed choice pairs must not be pooled into one global mode
    mixed = ([{"gold": "left", "choices": ["left", "right"]}] * 5
             + [{"gold": "superior", "choices": ["inferior", "superior"]}] * 5)
    tb2 = trivial_baseline(mixed)
    assert abs(tb2["per_pair_majority"] - 1.0) < 1e-9, tb2
    assert abs(tb2["random"] - 0.5) < 1e-9, tb2

    print("selftest OK — categorical extraction, normalisation, and an "
          "imbalance-aware chance level (majority class, not a flat 50%)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
