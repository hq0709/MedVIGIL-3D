"""
Score the target-contrast pairs: does the image tell the model which structure?

The construction (build_target_pairs.py) holds the lesion and the growth amount
fixed and varies only which structure is named, so the two members of a pair
carry identical numbers and differ only anatomically. Ordering numbers cannot
produce a differentiated answer.

Why the obvious metric is wrong
-------------------------------
The natural score -- fraction of pairs answered differently *and* correctly --
has a chance level of 25 %, not 0: two independent coin flips agree with two
opposite golds a quarter of the time. Worse, the blind arm cannot serve as the
null, because every model audited answers the blind probes with a single
constant token. Blind differentiation is therefore 0 % by degeneracy rather
than by inability, and a sighted-minus-blind "image gain" computed against it
credits the image for merely moving a model off a constant response.

What is measured instead
------------------------
Two quantities, reported separately because they answer different questions:

  differentiation rate -- how often the model gives the pair two different
    answers at all. This is a response-distribution property; the image moving
    it off zero says the image is being used as *something*.

  direction accuracy -- among the pairs it did differentiate, how often the
    "yes" landed on the member whose gap is under the growth amount. This is
    the grounding question, and its null is 50 % exactly, by construction and
    independent of how often the model chooses to differentiate.

Direction accuracy is tested against 50 % with an exact two-sided binomial test
and Holm-corrected across the models tested, since a per-model p just under .05
across several models is not evidence of anything.
"""
from __future__ import annotations

import os
from pathlib import Path

import argparse
import json
import math
from collections import Counter, defaultdict


def load(path: str) -> dict[str, list[dict]]:
    pairs = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        pairs[r["pair_id"]].append(r)
    return {k: v for k, v in pairs.items() if len(v) == 2}


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def arm_stats(pairs: dict[str, list[dict]]) -> dict:
    dist = Counter(x["prediction"] for v in pairs.values() for x in v)
    differ = [v for v in pairs.values()
              if v[0]["prediction"] != v[1]["prediction"]]
    correct = sum(all(x["prediction"] == x["gold"] for x in v) for v in differ)
    lo, hi = wilson(correct, len(differ))
    return {"n_pairs": len(pairs), "dist": dict(dist),
            "differ": len(differ), "differ_rate": 100 * len(differ) / len(pairs),
            "correct": correct, "n_differ": len(differ),
            "direction": 100 * correct / len(differ) if differ else float("nan"),
            "ci": [lo, hi], "p": binom_two_sided(correct, len(differ))}


def holm(ps: dict[str, float]) -> dict[str, float]:
    live = {k: v for k, v in ps.items() if v == v}          # drop NaN
    order = sorted(live, key=live.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, (len(order) - i) * live[k])
        out[k] = min(1.0, running)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("MEDVIGIL3D_ROOT", str(Path(__file__).resolve().parent.parent)))
    ap.add_argument("--models", nargs="+",
                    default=["qwen7b", "qwen32b", "internvl", "qwen3vl",
                             "internvl14", "m3d"])
    args = ap.parse_args()

    res = {}
    for m in args.models:
        try:
            res[m] = {a: arm_stats(load(f"{args.root}/tp_{m}_{a}.jsonl"))
                      for a in ("sighted", "blind")}
        except FileNotFoundError:
            print(f"{m}: not run")

    adj = holm({m: r["sighted"]["p"] for m, r in res.items()})

    print(f"{'model':10} {'arm':8} {'responses':24} {'differentiates':>15} "
          f"{'direction acc':>15} {'95% CI':>16} {'p':>8} {'Holm p':>8}")
    for m, r in res.items():
        for a in ("sighted", "blind"):
            s = r[a]
            d = "n/a" if s["n_differ"] == 0 else \
                f"{s['correct']}/{s['n_differ']} {s['direction']:.1f}%"
            ci = "n/a" if s["n_differ"] == 0 else \
                f"[{s['ci'][0]:.1f},{s['ci'][1]:.1f}]"
            p = "n/a" if s["n_differ"] == 0 else f"{s['p']:.4f}"
            hp = f"{adj[m]:.3f}" if a == "sighted" and m in adj else ""
            print(f"{m:10} {a:8} {str(s['dist'])[:24]:24} "
                  f"{s['differ_rate']:14.1f}% {d:>15} {ci:>16} {p:>8} {hp:>8}")
        r["holm_p"] = adj.get(m)

    print("\nchance direction accuracy is 50.0% by construction, independent of "
          "how\noften the model differentiates. Blind differentiation is 0% "
          "because every\nmodel answers the blind probes with one constant "
          "token, so it measures\ndegeneracy, not the absence of an anatomical "
          "prior -- which is why no\nsighted-minus-blind gain is reported here.")
    json.dump(res, open(f"{args.root}/target_pairs_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
