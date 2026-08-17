"""
Qwen2.5-VL on DeepTumorVQA, sighted vs blind -- the control that separates
DIMENSIONALITY from QUESTION TYPE.

The confound being removed
--------------------------
Measured so far with Qwen2.5-VL:
    VQA-RAD (native 2D radiographs, findings questions):   +12.7 pp from the image
    our 3D geometry QA (rendered volumes, geometric questions): -5.7 pp

Two things differ at once, so neither explains the gap on its own. DeepTumorVQA
holds the question type roughly fixed -- recognition, measurement, findings, the
same family VQA-RAD asks -- while keeping the input volumetric. Running the SAME
model here isolates which factor matters:

    image gain collapses here  -> dimensionality is the driver
    image gain holds up here   -> question type is the driver

The volumes are read from the cache written by fetch_dtvqa.py (already resampled
to 32x256x256) and rendered to orthogonal views, so no re-download is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from render import montage, orthogonal_views  # noqa: E402


def render_cached(npy_path: Path) -> np.ndarray:
    """Cached volume (1, 32, 256, 256) in [0,1] -> orthogonal-view montage.

    The cache is already min-max normalised, so windowing is skipped; scaling to
    uint8 directly preserves whatever contrast the M3D preprocessing produced,
    which is what makes this comparable to the M3D run on the same cache.
    """
    vol = np.load(npy_path).astype(np.float32)[0]        # (32, 256, 256) = (z,y,x)
    vol = np.transpose(vol, (2, 1, 0))                   # -> (x, y, z) for render
    img = (np.clip(vol, 0, 1) * 255).astype(np.uint8)
    # the cache is isotropic in index space; spacing 1 keeps it that way
    r = orthogonal_views(img.astype(np.int16), np.array([1.0, 1.0, 1.0]),
                         preset="soft_tissue", isotropic=False)
    return montage(r.available())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir from fetch_dtvqa.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
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

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    print("model ready", flush=True)

    def score(question: str, opts: dict, img) -> tuple:
        listing = " ".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": (f"This is a CT scan shown as axial, coronal and sagittal "
                      f"views. {question} {listing} "
                      f"Answer with the option text only.")}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        out = {}
        for letter, txt in sorted(opts.items()):
            enc = proc(text=[text + txt], images=[img],
                       return_tensors="pt").to(args.device)
            n_c = len(proc.tokenizer(txt, add_special_tokens=False)["input_ids"])
            with torch.inference_mode():
                logits = model(**enc).logits[0, :-1].float()
            tgt = enc["input_ids"][0, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            out[letter] = float(lp[-n_c:].gather(1, tgt[-n_c:, None]).mean())
        return max(out, key=out.get), out

    written = errors = 0
    with open(args.out, "w") as f:
        for vid, items in sorted(by_vol.items()):
            npy = data / "vol_cache" / f"{vid}.npy"
            if not npy.exists():
                continue
            arr = render_cached(npy)
            img = Image.fromarray(arr).convert("RGB")
            if args.blind:
                img = Image.new("RGB", img.size, (128, 128, 128))
            for r in items:
                try:
                    letter, sc = score(r["question"], r["options"], img)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {r['qid']}: {e}", file=sys.stderr)
                    continue
                f.write(json.dumps({
                    "qid": r["qid"], "pred_option": letter,
                    "correct_option": r["correct_option"],
                    "question_type": r["question_type"],
                    "question_subtype": r["question_subtype"],
                    "logprobs": {k: round(v, 4) for k, v in sc.items()},
                }) + "\n")
                written += 1
            f.flush()
            print(f"  {vid}: {len(items)} answered", flush=True)

    print(f"\nwrote {written} (errors: {errors})")


def selftest() -> None:
    import tempfile

    vol = np.zeros((1, 32, 256, 256), dtype=np.float32)
    vol[0, 24:, :, :] = 1.0                       # bright band at high z
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, vol)
        m = render_cached(Path(f.name))

    assert m.ndim == 2 and m.dtype == np.uint8, (m.shape, m.dtype)
    assert m.max() > 200, m.max()
    # a montage of three views must be wider than any single one
    assert m.shape[1] > m.shape[0], m.shape
    print(f"selftest OK — cached volume renders to a {m.shape} uint8 montage "
          f"with the bright band preserved (max {m.max()})")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
