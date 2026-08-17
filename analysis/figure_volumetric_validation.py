"""
Figure 6 -- volumetric validation.

The only figure in the journal version that does not already exist. It is drawn
in the conference paper's house style (Comic Sans stack, 8 pt base, provider
palette, type-42 embedding) so it sits beside the five reused 2D figures without
looking like it came from somewhere else.

Inputs are regenerated from raw by from3d/make_figure_data.py -- NOT the stale
CSVs that shipped in the archive. Two of the thirteen rows in the shipped
fig7/fig8 (the two M3D models) were from an earlier run and disagreed with the
manuscript; regenerating fixes them and reproduces the published ranges exactly
(48.7-51.0 %, image contribution -1.3 to +1.0 pp, 13/13 CIs containing 50).

Two panels, because the volumetric result has two halves that must be read
together:

  (a) forest plot of confound-free accuracy with 95 % CIs against the 50 % line.
      Every interval crosses 50, so the panel's message is the absence of any
      model clearing chance -- legible only if the chance line is the visual
      anchor rather than an afterthought.

  (b) the known-answer response control. Panel (a) is uninterpretable for a model
      that has no working response channel, so the control belongs beside it
      rather than in a supplement. Bars are the number of the seven control
      questions answered correctly on all twenty volumes; the annotation is the
      model's yes-rate, which should sit at 42.9 % because three of the seven
      questions have gold "yes". The four failing models miss it in both
      directions -- Med3DVLM answers "no" to everything, M3D-LaMed-Llama2 "yes"
      to most things.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "from3d" / "figdata"
OUT = BASE / "figures"
FONT_DIR = BASE.parent / "MedVIGIL_NeurIPS2026" / "fonts" / "mscorefonts"

# --- conference-paper house style -------------------------------------------
for fp in (FONT_DIR / "Comic.TTF", FONT_DIR / "Comicbd.TTF"):
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Comic Sans MS", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Provider palette, extended from the conference figures. Qwen and the medical
# systems keep the exact hues they carry in the 2D panels so a reader tracking a
# family across figures sees the same colour.
PROVIDER_COLOR = {
    "Qwen":       "#915dff",   # as in the 2D figures
    "InternVL":   "#10a37f",
    "Mistral":    "#d97757",
    "HuggingFace": "#4285f4",
    "LLaVA":      "#a83279",   # LLaVA family, as in the 2D figures
    "Medical 3D": "#c0392b",   # native volumetric medical VLMs
}
PROVIDER = {
    "SmolVLM2-2.2B": "HuggingFace", "Idefics3-8B": "HuggingFace",
    "Qwen2.5-VL-3B": "Qwen", "Qwen2.5-VL-7B": "Qwen",
    "Qwen2.5-VL-32B": "Qwen", "Qwen3-VL-8B": "Qwen",
    "InternVL3-8B": "InternVL", "InternVL3-14B": "InternVL",
    "LLaVA-OneVision-7B": "LLaVA", "Pixtral-12B": "Mistral",
    "M3D-LaMed-Phi3-4B": "Medical 3D", "M3D-LaMed-Llama2-7B": "Medical 3D",
    "Med3DVLM-7B": "Medical 3D",
}

# A model fails the control if it answers fewer than five of the seven
# known-answer questions correctly on every volume.
PERFECT_FAIL_MAX = 4
EXPECTED_YES = 42.9   # three of seven control questions have gold "yes"
GREY = "#999999"


def load():
    acc = {r["model"]: r for r in
           csv.DictReader(open(FIG / "fig7_forest_confound_free.csv"))}
    ctl = {r["model"]: r for r in
           csv.DictReader(open(FIG / "fig8_response_controls.csv"))}
    rows = []
    for m, a in acc.items():
        c = ctl[m]
        perfect = int(c["perfect_questions"])
        rows.append({
            "model": m, "input": a["input"],
            "acc": float(a["acc"]), "lo": float(a["ci_lo"]), "hi": float(a["ci_hi"]),
            "gain": float(a["image_gain"]),
            "perfect": perfect, "n_q": int(c["n_questions"]),
            "yes_rate": float(c["yes_rate"]),
            "failed": perfect <= PERFECT_FAIL_MAX,
        })
    # native systems at the top, then montage models by accuracy
    rows.sort(key=lambda r: (r["input"] == "native", r["acc"]))
    return rows


def main():
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    y = list(range(len(rows)))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 5.2), sharey=True,
        gridspec_kw={"width_ratios": [2.45, 1.0], "wspace": 0.06})

    # ---- panel (a): confound-free accuracy ------------------------------
    ax1.axvspan(49.0, 51.0, color="#f2f2f2", zorder=0)
    ax1.axvline(50, color=GREY, lw=1.3, ls="--", zorder=1)
    ax1.text(50, len(rows) - 0.35, " chance = 50%", color="#777777",
             fontsize=7.6, va="bottom", ha="left")
    for i, r in enumerate(rows):
        col = PROVIDER_COLOR[PROVIDER[r["model"]]]
        ax1.plot([r["lo"], r["hi"]], [i, i], color=col, lw=1.5,
                 solid_capstyle="round", zorder=2, alpha=0.9)
        for xb in (r["lo"], r["hi"]):
            ax1.plot([xb, xb], [i - .15, i + .15], color=col, lw=1.1, zorder=2)
        ax1.scatter(r["acc"], i, s=46, color=col, edgecolor="white",
                    linewidth=1.0, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [r["model"] + ("  †" if r["failed"] else "") for r in rows], fontsize=8)
    ax1.set_xlabel("Accuracy on confound-free probes (%),  95% CI",
                   fontsize=9, labelpad=4)
    ax1.set_xlim(44.5, 55.5)
    ax1.set_ylim(-0.7, len(rows) - 0.25)
    ax1.set_title("(a)  Every interval contains chance", fontsize=9.5,
                  pad=6, loc="left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", labelsize=7.5)

    # ---- panel (b): response-channel control ----------------------------
    for i, r in enumerate(rows):
        col = PROVIDER_COLOR[PROVIDER[r["model"]]]
        ax2.barh(i, r["perfect"], height=0.5, color=col,
                 alpha=0.35 if r["failed"] else 0.9,
                 edgecolor=col, linewidth=1.0, zorder=2)
        if r["failed"]:
            ax2.text(r["perfect"] + 0.18, i, f"{r['yes_rate']:.0f}% yes",
                     fontsize=7, va="center", color="#a03030", zorder=4)
    ax2.axvline(r["n_q"], color=GREY, lw=1.0, ls=":", zorder=1)
    ax2.set_xlabel("Control questions answered\ncorrectly on every volume  (of 7)",
                   fontsize=9, labelpad=4)
    ax2.set_xlim(0, 9.6)
    ax2.set_xticks([0, 2, 4, 6, 7])
    ax2.set_title("(b)  Can the model answer at all?", fontsize=9.5,
                  pad=6, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", labelsize=7.5)

    n_native = sum(r["input"] == "native" for r in rows)
    if n_native:
        split = len(rows) - n_native - 0.5
        for ax in (ax1, ax2):
            ax.axhline(split, color="#cccccc", lw=0.9, zorder=1)
        ax1.text(55.35, split + 0.14, "native volumetric", fontsize=7.4,
                 color="#888888", va="bottom", ha="right", style="italic")

    handles = [Patch(facecolor=c, edgecolor="white", label=k)
               for k, c in PROVIDER_COLOR.items()]
    handles.append(Line2D([], [], color="none",
                          label="†  fails response-channel control"))
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7.3,
               frameon=False, bbox_to_anchor=(0.5, -0.035))

    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        p = OUT / f"fig6_volumetric_validation.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  wrote {p}")

    n50 = sum(r["lo"] <= 50 <= r["hi"] for r in rows)
    fails = [r["model"] for r in rows if r["failed"]]
    print(f"\n{n50}/{len(rows)} CIs contain 50%")
    print(f"accuracy {min(r['acc'] for r in rows):.1f}-{max(r['acc'] for r in rows):.1f} %, "
          f"image contribution {min(r['gain'] for r in rows):+.1f} to "
          f"{max(r['gain'] for r in rows):+.1f} pp")
    print(f"{len(fails)}/{len(rows)} fail the response-channel control: {', '.join(fails)}")


if __name__ == "__main__":
    main()
