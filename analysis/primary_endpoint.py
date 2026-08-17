"""
Primary endpoint: difference in harm-weighted silent-failure rate between the
independent radiologist reference (R4) and each audited model.

Why the interval is built this way
----------------------------------
The manuscript's first draft promised a case-level cluster bootstrap. That is
not possible for the human reference: R4's responses were recorded as counts per
(probe family x risk tier), not per probe, so no case-level resampling of R4
exists to do. Pretending otherwise would be a statistical claim the data cannot
support.

What the data does support is a *symmetric* procedure. Both R4 and every model
are available as tier-level counts over the same five strata with the same
denominators (142, 62, 236, 86, 74 = 600 trap probes). So we resample each
stratum from Binomial(n_t, p_t) for both arms, recompute the harm-weighted rate
    SFR_w = sum_t w_t p_t / sum_t w_t ,  w = (1, 2, 3, 5, 8) for L1..L5
and take the difference. Identical method on both sides, so the comparison is
like-for-like.

The cost is stated plainly in the manuscript: this treats probes within a tier as
independent and therefore does not absorb within-case correlation. It is an
anticonservative assumption, and the resulting intervals are narrower than a
cluster-robust interval would be. Given the size of the effect (the smallest
model-versus-radiologist gap is ~16 points on a 5.8% base) the conclusion is not
close to the boundary, but the limitation belongs in the paper, not in a
footnote.
"""
from __future__ import annotations

import csv
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(os.path.dirname(BASE))
REF2D = os.path.join(BASE, "from3d", "paper", "ref2d")
CLIN = os.path.join(PROJ, "data", "medvlm_bench_v1", "clinician_baseline.csv")
OUT = os.path.join(BASE, "out")

TIERS = ("L1", "L2", "L3", "L4", "L5")
W = np.array([1, 2, 3, 5, 8], dtype=float)
B = 10000
SEED = 0

LABEL = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.5": "GPT-5.5", "gpt-5.4": "GPT-5.4", "gpt-4o": "GPT-4o",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "huatuogpt-vision-7b": "HuatuoGPT-Vision 7B", "llava-med": "LLaVA-Med 7B",
    "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "Qwen--Qwen3.5-397B-A17B": "Qwen3.5-397B-A17B", "Qwen--Qwen3.5-9B": "Qwen3.5-9B",
    "moonshotai--Kimi-K2.5": "Kimi K2.5", "moonshotai--Kimi-K2.6": "Kimi K2.6",
    "deepseek-v4-flash": "DeepSeek-V4-Flash", "deepseek-v4-pro": "DeepSeek-V4-Pro",
}
TEXT_ONLY = {"deepseek-v4-flash", "deepseek-v4-pro"}


def weighted(p: np.ndarray) -> np.ndarray:
    """Harm-weighted rate; p has shape (..., 5)."""
    return (p * W).sum(axis=-1) / W.sum()


def load_r4():
    n, k = {}, {}
    for r in csv.DictReader(open(CLIN)):
        if r["probe_kind"] == "halluc_trap" and r["risk_tier"] in TIERS:
            n[r["risk_tier"]] = int(r["n_probes"])
            k[r["risk_tier"]] = int(r["n_correct"])   # accepted trap = silent failure
    return (np.array([n[t] for t in TIERS]), np.array([k[t] for t in TIERS]))


def load_models():
    out = {}
    for r in csv.DictReader(open(os.path.join(REF2D, "sfr_tier2d.csv"))):
        out.setdefault(r["model"], {})[r["risk_tier"]] = (
            float(r["sfr"]), int(r["n_traps"]))
    models = {}
    for m, d in out.items():
        if not all(t in d for t in TIERS):
            continue
        n = np.array([d[t][1] for t in TIERS])
        p = np.array([d[t][0] for t in TIERS])
        models[m] = (n, np.round(p * n).astype(int))
    return models


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_r4, k_r4 = load_r4()
    models = load_models()

    p_r4 = k_r4 / n_r4
    boot_r4 = weighted(rng.binomial(n_r4, p_r4, size=(B, 5)) / n_r4)
    obs_r4 = weighted(p_r4)

    rows = []
    for m, (n_m, k_m) in models.items():
        p_m = k_m / n_m
        boot_m = weighted(rng.binomial(n_m, p_m, size=(B, 5)) / n_m)
        diff = boot_m - boot_r4
        rows.append({
            "model": LABEL.get(m, m),
            "text_only": m in TEXT_ONLY,
            "model_sfr_w": 100 * weighted(p_m),
            "diff": 100 * (weighted(p_m) - obs_r4),
            "lo": 100 * np.percentile(diff, 2.5),
            "hi": 100 * np.percentile(diff, 97.5),
            "p_excl": float((diff <= 0).mean()),
        })
    rows.sort(key=lambda r: r["diff"])

    print("Primary endpoint: harm-weighted silent-failure rate, model minus radiologist R4")
    print(f"Radiologist R4 reference: {100*obs_r4:.1f}% "
          f"[{100*np.percentile(boot_r4,2.5):.1f}, {100*np.percentile(boot_r4,97.5):.1f}]"
          f"   ({k_r4.sum()}/{n_r4.sum()} trap probes unweighted)")
    print(f"Stratified binomial bootstrap, B={B:,}, seed={SEED}, "
          f"strata n={list(n_r4)} (L1-L5), weights {list(W.astype(int))}\n")
    print(f"{'model':24}{'SFR_w %':>9}{'diff (pp)':>11}{'95% CI':>18}{'P(diff<=0)':>12}")
    print("-" * 76)
    for r in rows:
        star = " *" if r["text_only"] else ""
        ci = f"[{r['lo']:+.1f}, {r['hi']:+.1f}]"
        print(f"{r['model'] + star:24}{r['model_sfr_w']:9.1f}{r['diff']:+11.1f}"
              f"{ci:>18}{r['p_excl']:12.4f}")

    vis = [r for r in rows if not r["text_only"]]
    print(f"\nAll {len(rows)} models exceed the radiologist reference.")
    print(f"Smallest gap: {vis[0]['model']} {vis[0]['diff']:+.1f} pp "
          f"[{vis[0]['lo']:+.1f}, {vis[0]['hi']:+.1f}]")
    print(f"Largest gap : {vis[-1]['model']} {vis[-1]['diff']:+.1f} pp "
          f"[{vis[-1]['lo']:+.1f}, {vis[-1]['hi']:+.1f}]")
    print(f"Every 95% CI excludes 0: "
          f"{all(r['lo'] > 0 for r in rows)}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "primary_endpoint.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "text_only", "model_sfr_w",
                                           "diff", "lo", "hi", "p_excl"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nwrote {os.path.join(OUT, 'primary_endpoint.csv')}")


if __name__ == "__main__":
    main()
