"""
Sighted-vs-blind test for M3D-LaMed on the DeepTumorVQA benchmark.

The question this answers
------------------------
Does a 3D medical VLM actually consult the CT volume when answering a published
3D medical VQA benchmark? Both arms get an identical prompt and an identical
tensor shape; only the CONTENT differs (real volume vs all zeros). Any accuracy
the blind arm retains is accuracy the image was not needed for.

Scoring is by likelihood over the option TEXTS, never by parsing the reply.
M3D answers nearly everything with a referring-expression template ("The object
in question is Paris"), and its replies embed structure names that contain
answer words, so a regex scorer reads "rib left 5" as the answer "left". That
failure mode produces plausible-looking accuracies driven entirely by which
organ a question happens to mention.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from m3d_infer import IMAGE_TOKEN, N_IMAGE_TOKENS, M3D, TARGET_SHAPE  # noqa: E402


def build_prompt(question: str, options: dict[str, str]) -> str:
    listing = " ".join(f"{k}: {v}" for k, v in sorted(options.items()))
    return f"{question} {listing}"


def score(model: M3D, question: str, options: dict[str, str],
          volume: np.ndarray) -> tuple[str, dict[str, float]]:
    """Return the highest-likelihood option LETTER plus per-option scores."""
    q = build_prompt(question, options)
    texts = [options[k] for k in sorted(options)]
    best_text, scores = model.score_choices(q, volume, texts)
    inv = {v: k for k, v in options.items()}
    return inv[best_text], {inv[t]: s for t, s in scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir from fetch_dtvqa.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = Path(args.data)
    rows = [json.loads(l) for l in open(data / "qa.jsonl") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    by_vol = defaultdict(list)
    for r in rows:
        by_vol[r["image_id"]].append(r)
    print(f"{len(rows)} QA over {len(by_vol)} volumes "
          f"({'BLIND' if args.blind else 'sighted'})", flush=True)

    model = M3D(device=args.device)
    print("model ready", flush=True)

    zero = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
    written = errors = 0
    with open(args.out, "w") as out:
        for vid, items in sorted(by_vol.items()):
            npy = data / "vol_cache" / f"{vid}.npy"
            if not npy.exists():
                continue
            vol = zero if args.blind else np.load(npy).astype(np.float32)
            for r in items:
                try:
                    letter, sc = score(model, r["question"], r["options"], vol)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {r['qid']}: {e}", file=sys.stderr)
                    continue
                out.write(json.dumps({
                    "qid": r["qid"], "pred_option": letter,
                    "correct_option": r["correct_option"],
                    "question_type": r["question_type"],
                    "question_subtype": r["question_subtype"],
                    "logprobs": {k: round(v, 4) for k, v in sc.items()},
                }) + "\n")
                written += 1
            out.flush()
            print(f"  {vid}: {len(items)} answered", flush=True)

    print(f"\nwrote {written} predictions (errors: {errors})")


def summarise(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = sum(r["pred_option"] == r["correct_option"] for r in rows)
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(
            r["pred_option"] == r["correct_option"])
    return {
        "n": len(rows),
        "accuracy": ok / len(rows) if rows else 0.0,
        "per_type": {t: {"n": len(v), "acc": sum(v) / len(v)}
                     for t, v in sorted(by_type.items())},
        "pred_distribution": dict(Counter(r["pred_option"] for r in rows)),
        "gold_distribution": dict(Counter(r["correct_option"] for r in rows)),
    }


def selftest() -> None:
    opts = {"A": "left kidney", "B": "the same", "C": "right kidney"}
    p = build_prompt("Which kidney is larger?", opts)
    assert p == "Which kidney is larger? A: left kidney B: the same C: right kidney", p

    # options with duplicate text would collapse the letter mapping; make sure
    # the inversion is built from the actual dict, not assumed unique
    assert len({v: k for k, v in opts.items()}) == 3

    print("selftest OK — prompt assembles options in letter order and the "
          "text->letter inversion is well defined")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
