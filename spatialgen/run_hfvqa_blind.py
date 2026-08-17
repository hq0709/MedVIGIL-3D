"""
Sighted-vs-blind measurement on any HuggingFace medical VQA dataset.

Purpose in the wider project
----------------------------
The counterfactual generator needs 3D volumes with lesion masks, so it cannot be
applied to 2D benchmarks like VQA-RAD or PathVQA -- a projection radiograph has
no well-defined "distance to the aorta". The METHOD, however, is dataset
agnostic, and that is where generality comes from: point this at any established
benchmark and it reports how much of that benchmark survives removing the image.

Scoring is by likelihood over the answer strings rather than by parsing free
text. Medical VLMs answer in fixed templates whose wording embeds candidate
answer words, so a regex scorer credits the model for text it did not choose.

The blind arm passes a mid-grey image of identical size rather than dropping the
image, keeping the architecture path and token count matched so the only
difference is content.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_items(dataset: str, split: str, limit: int | None,
               binary_only: bool, seed: int = 0) -> list[dict]:
    import random

    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    cols = ds.column_names
    q_col = next(c for c in ("question", "Question", "query") if c in cols)
    a_col = next(c for c in ("answer", "Answer", "label") if c in cols)
    i_col = next(c for c in ("image", "Image", "img") if c in cols)

    items = []
    for idx, r in enumerate(ds):
        ans = str(r[a_col]).strip().lower()
        if binary_only and ans not in ("yes", "no"):
            continue
        items.append({"qid": f"{idx}", "question": str(r[q_col]),
                      "answer": ans, "image": r[i_col],
                      "choices": ["no", "yes"] if ans in ("yes", "no") else None})
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def trivial_baseline(items: list[dict]) -> dict:
    """Best score without reading the question: random, or the majority label."""
    c = Counter(i["answer"] for i in items)
    n = len(items)
    rand = sum(1 / len(i["choices"]) for i in items) / n if n else 0.0
    maj = max(c.values()) / n if n else 0.0
    return {"random": rand, "majority": maj, "reference": max(rand, maj),
            "label_counts": dict(c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flaviagiammarino/vqa-rad")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--binary-only", action="store_true", default=True)
    args = ap.parse_args()

    items = load_items(args.dataset, args.split, args.limit, args.binary_only)
    tb = trivial_baseline(items)
    print(f"{args.dataset} [{args.split}]: {len(items)} binary items "
          f"({'BLIND' if args.blind else 'sighted'})")
    print(f"  labels {tb['label_counts']}  trivial baseline {tb['reference']:.1%}",
          flush=True)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    print("model ready", flush=True)

    def score(question: str, img: Image.Image, choices: list[str]) -> tuple:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": f"{question} Answer with exactly one of: "
                     f"{', '.join(choices)}."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        out = {}
        for c in choices:
            full = text + c
            enc = proc(text=[full], images=[img], return_tensors="pt").to(args.device)
            n_c = len(proc.tokenizer(c, add_special_tokens=False)["input_ids"])
            with torch.inference_mode():
                logits = model(**enc).logits[0, :-1].float()
            tgt = enc["input_ids"][0, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            out[c] = float(lp[-n_c:].gather(1, tgt[-n_c:, None]).mean())
        best = max(out, key=out.get)
        return best, out

    written = errors = 0
    with open(args.out, "w") as f:
        for i, it in enumerate(items, 1):
            img = it["image"].convert("RGB")
            if args.blind:
                # matched-size neutral image: same vision path, no content
                img = Image.new("RGB", img.size, (128, 128, 128))
            try:
                pred, sc = score(it["question"], img, it["choices"])
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  failed {it['qid']}: {e}", file=sys.stderr)
                continue
            f.write(json.dumps({"qid": it["qid"], "prediction": pred,
                                "gold": it["answer"],
                                "logprobs": {k: round(v, 4)
                                             for k, v in sc.items()}}) + "\n")
            written += 1
            if i % 50 == 0:
                f.flush()
                print(f"  {i}/{len(items)}", flush=True)

    print(f"\nwrote {written} (errors: {errors})")
    print(json.dumps({"trivial_baseline": tb}, indent=1))


def summarise(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = sum(r["prediction"] == r["gold"] for r in rows)
    return {"n": len(rows), "accuracy": ok / len(rows) if rows else 0.0,
            "pred_dist": dict(Counter(r["prediction"] for r in rows)),
            "gold_dist": dict(Counter(r["gold"] for r in rows))}


def selftest() -> None:
    items = [{"answer": "yes", "choices": ["no", "yes"]}] * 6 + \
            [{"answer": "no", "choices": ["no", "yes"]}] * 4
    tb = trivial_baseline(items)
    assert abs(tb["random"] - 0.5) < 1e-9, tb
    assert abs(tb["majority"] - 0.6) < 1e-9, tb
    assert abs(tb["reference"] - 0.6) < 1e-9, tb

    # a perfectly balanced set must fall back to the 50% random level, not to a
    # spuriously high majority
    bal = [{"answer": "yes", "choices": ["no", "yes"]}] * 5 + \
          [{"answer": "no", "choices": ["no", "yes"]}] * 5
    assert abs(trivial_baseline(bal)["reference"] - 0.5) < 1e-9

    print("selftest OK — trivial baseline takes the max of random and majority, "
          "and does not inflate on a balanced set")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
