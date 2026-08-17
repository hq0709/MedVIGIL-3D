# Experiments to run

Tiered by whether they gate the journal submission. **Tier 0 cannot be skipped.**
Tier 1 kills a likely reviewer objection or materially strengthens the paper.
Tier 2 is optional.

Read the dependency graph before starting — one ordering constraint really matters.

## Setup

```bash
export MEDVIGIL3D_ROOT=$PWD
export MSD_ROOT=/path/to/MSD          # medicaldecathlon.com, Tasks 03/06/07/10
```

You also need `cfqa_*/seg_cache/` (TotalSegmentator masks, 593 files / 544 MB) —
not in this repo; regenerate with `spatialgen/run_pipeline.py` or copy it over.

**On GPUs.** `MontageModel` loads in bf16 with `device_map=device` and scores each
answer option with a **separate, unbatched forward pass**. So VRAM is dominated by
weights (≈ params × 2 bytes, plus KV cache and activations) and a single process
will not saturate a large card's compute. Throughput comes from **running many
processes in parallel across GPUs**, not from bigger batches. `--device auto`
shards one model across all visible GPUs — required for the 32B and 25B-MoE tags.

| tag | params | bf16 weights ≈ |
|---|---|---|
| `qwen32b` | 32B | 64 GB → `--device auto` or one 80 GB card |
| `aria` | 25B MoE | 50 GB |
| `internvl14` | 14B | 28 GB |
| `pixtral` | 12B | 24 GB |
| `internvl`, `qwen3vl`, `idefics3` | 8B | 16 GB |
| `qwen7b`, `llavaov` | 7B | 14 GB |
| `m3d`, `m3dllama`, `med3dvlm` | 4–7B | 8–14 GB + a 32×256×256 volume tensor |

---

# Tier 0 — gating

## E1. Identification control

**Question.** Does the headline null survive telling the model *which* lesion and
*which* target the question is about?

**Why it gates.** Every published volumetric result fed the model
`montage(orthogonal_views(volume))`. In `render.orthogonal_views`, `index=None`
renders **three slices through the geometric centre of the volume, unannotated**.
The question asks about "**this lesion**" while a qid like
`liver_0_lesion5_heart_g5.8` shows that liver holds at least five lesions. So
nothing identifies the lesion, nothing marks the target, the lesion need not appear
in any of the three slices, and no scale bar is drawn although the question is
metric. **The null therefore admits a mundane reading that says nothing about
geometric reasoning: the probe may be underspecified given the model's input.** A
reviewer raises this in one sentence and the manuscript has no answer.

The existing reader study does not settle it — `render_case` gives the radiologist
lesion-in-red, target-in-cyan, joint-visibility slices and a 10 mm scale bar, i.e.
strictly more information than any model received.

**Design.** `spatialgen/run_identification_control.py`, four conditions:

| `--condition` | slices | outlines | scale bar | isolates |
|---|---|---|---|---|
| `plain` | volume centre | no | no | the published condition (paired control) |
| `bestslice` | joint-visibility | no | no | does it need to **see** the structures? |
| `overlay` | volume centre | red / cyan | no | does it need to know **which** structures? |
| `identified` | joint-visibility | red / cyan | yes | exactly what the reader sees |

Panel geometry and colours match `render_case`, so `identified` is directly
comparable with the reader study. The prompt gains a legend sentence **only** when
those pixels are actually drawn.

**Scope.** All four conditions × 4 organs × 4 models
(`qwen32b`, `internvl`, `qwen3vl`, `qwen7b`). Include `qwen7b` — it answered "no"
to all 2,262 probes, so it is the cleanest test of whether identification unsticks
a degenerate responder. Matched subset only (`--subset matched`).

= 64 runs. Start with the minimum viable cell:

```bash
for COND in plain bestslice overlay identified; do
  python spatialgen/run_identification_control.py \
    --qa        cfqa_Task03_Liver/qa \
    --task-dir  $MSD_ROOT/Task03_Liver \
    --seg-cache cfqa_Task03_Liver/seg_cache \
    --model     qwen32b --condition $COND --subset matched \
    --device    auto \
    --out       id_Task03_Liver_qwen32b_$COND.jsonl
done
```

Verify with `--limit 20` first. Liver has the most lesions per volume, so it is
where the ambiguity bites hardest — run it first.

**Cost.** 1,368 probes on the matched subset; renders cached per (lesion, target)
pair, so ~684 renders and 2 forward passes per probe. Tens of minutes per
model-condition on one GPU, not hours.

**Score with** `python analysis/image_contribution_ci.py` (volume-level paired
bootstrap; output format matches `mm_*`/`roi_*` so it reads unchanged).

**Outcome → claim.**

| result | what the paper must say |
|---|---|
| `identified` CI still contains 50 % | The null is about geometric reasoning and is now defensible: the model got everything the reader gets and still could not answer. Report the control; the 3D section gets materially stronger. |
| `identified` CI excludes 50 % upward | The published null was substantially an input-specification artefact. Weaken the claim and rewrite Results. Far better found here than in review. |
| `bestslice` or `overlay` alone recovers most of the gain | Name the missing cue. That is a finding about how volumetric probes must be presented, publishable in its own right. |

**Lands in.** Headline if it changes the result; supplement + a defensive paragraph
if it does not.

## E2. Task-solvability ceiling

**Question.** Is the probe answerable at all, by anything?

**Why it gates.** "All models at chance" is only interpretable against a
demonstrated ceiling. Without one, a reviewer can say the task is ill-posed rather
than hard. Two cheap arms bound it from above:

- **`text-oracle`** — supply `gap_mm` and `growth_mm` in the prompt, no image. The
  answer is then the numeric comparison `growth ≥ gap`. A model at chance *here* is
  failing arithmetic, not vision, which reframes every other result.
- **`geometry-oracle`** — score the reference rule directly from the stored
  provenance. Must be exactly 100 % by construction; it verifies the harness, not
  the model.

Both are seconds of compute and no new rendering. Implement as extra conditions on
the same runner.

**Outcome → claim.** If `text-oracle` is near ceiling, the arithmetic is available
and the failure is in getting geometry out of the image — which is the paper's
claim, now with a ceiling to point at. If `text-oracle` is also at chance, the
paper's framing must change substantially.

**Lands in.** One sentence in Results plus a supplement row. Cheap insurance.

---

# Tier 1 — strengthening

## E3. Sub-task decomposition — where does the failure live?

**Question.** Can the model do the components even when it fails the composite?

Four independent probes on the same volumes, each a separate condition:

1. **Localise** — "Is the outlined lesion in the liver, the lung, or the pancreas?"
   (tests that the overlay is perceived at all)
2. **Name the target** — "Is the cyan structure the aorta, the heart, or the
   sternum?" (tests target identification)
3. **Estimate distance** — "Approximately how many mm separate the two outlined
   structures? Choose the nearest: 5 / 15 / 30 / 60." (tests metric perception)
4. **Compare** — given the distance in text, answer the contact question
   (= E2 `text-oracle`)

A model that passes 1–3 and fails the composite has a *composition* failure; one
that fails 3 has a *metric perception* failure; one that fails 1–2 has a *binding*
failure. This turns "models fail" into "models fail here", which is the difference
between a null result and a mechanism.

**Scope.** 4 sub-tasks × 300 probes × 4 models. Forward passes only.

**Lands in.** Supplement, and one Discussion sentence. This is the single highest
scientific-value item after E1.

## E4. Scale — does the null survive?

**Question.** Is chance-level performance a property of current models or of the
sizes tested?

Currently the audit spans 2.2B → 32B, all at chance. That is a strong claim already,
but the obvious rebuttal is "you did not try the big ones."

Add the largest open VLMs and rerun the matched subset under `plain` and
`identified`. Candidates — **verify availability and add to `MODEL_ID` in
`run_multimodel.py` before running**:

| model | params | bf16 ≈ | note |
|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | 72B | 145 GB | needs `--device auto` across ≥2×80 GB |
| InternVL3-78B | 78B | 156 GB | same |
| Llama-3.2-90B-Vision-Instruct | 90B | 180 GB | same |

This is where idle capacity should go. Each is a long-running, VRAM-saturating job.

**Outcome → claim.** Null holds at 72–90B → "the failure does not close with scale
over a 40× range", a much stronger sentence than the current one. Null breaks →
report the scale at which it breaks; that is a headline finding either way.

**Lands in.** Headline (one row added to the existing table).

## E5. Inference-time compute

**Question.** Does letting the model reason, or sampling it repeatedly, recover the
signal?

The 2D half already ships chain-of-thought and self-verification ablations
(`results/ablation/*__cot.jsonl`, `*__sysverify.jsonl`), so this axis has precedent
and a reviewer will expect it in 3D too.

Arms, on the matched subset, `identified` condition, 2 models:

- greedy (baseline)
- chain-of-thought, then extract the answer
- self-consistency, k = 5 and k = 10, majority vote
- self-verification (answer, then critique own answer, then re-answer)

k = 10 sampling is 10× the forward passes — genuinely compute-hungry, and it
parallelises trivially across GPUs.

**Outcome → claim.** No recovery → "the failure is not a decoding or effort
problem", which pre-empts a standard objection. Recovery → an important positive
result about how volumetric probes must be run.

**Lands in.** Supplement plus one Results sentence.

## E6. Native volumetric models — repair or characterise

**Question.** Do the three released native volumetric medical VLMs have a usable
response channel at all, and does giving them the volume (rather than a montage)
help once they do?

All three fail the known-answer control, so their chance-level rows are currently
uninterpretable. Two things to try:

1. **Prompt-format repair.** `run_multimodel.py` already notes that applying the
   montage framing to a native model cost M3D-LaMed-Phi3 41 points on the
   response-channel controls. Re-run the controls with native framing, per model,
   and report whether the channel recovers. `sanity_*_montageprompt.jsonl` files
   exist for comparison.
2. **If a channel recovers**, run that model on the matched subset with the true
   volume, both `plain` and `identified` equivalents.

**Outcome → claim.** Channel recovers and still at chance → the strongest version
of the native-model finding. Channel does not recover → say plainly that these
released models cannot be evaluated as shipped, which is itself a finding a
radiology audience should hear.

**Lands in.** Headline sentence + supplement table.

## E7. Reader study — input-matched

**Question.** Can a radiologist do this task from the same input the model gets, and
are these probes clinically meaningful?

`reader_study/` holds a 104-probe form and answer key; **every `reader_answer` cell
is empty**. Regenerate images with `spatialgen/export_reader_study.py`.

Do this **after E1**, and make it input-matched: readers should see the
`identified` rendering, which is what `render_case` already produces. Otherwise a
reader-versus-model number compares two different inputs and means nothing.

Design upgrades worth making:

- **≥ 3 readers**, so inter-reader agreement can be reported (Fleiss' κ or
  Gwet's AC1). One reader gives no variability estimate — the current 2D limitation.
- Use the form's existing `clinical_relevance__1to5` column. If readers rate these
  probes 4–5, the whole probe design is justified against the "is this clinically
  meaningful?" objection. If they rate them 1–2, that is important to know before
  publication.
- Keep `severity_agree__yes_no` to validate the risk tiers.

**Not a blocker** — the external review confirmed this, *conditional on* the
volumetric section never claiming a clinician-validated reference standard. The
manuscript currently calls it *external testing* and states plainly that no reader
confirmation was performed. Keep that wording and the paper stands without this.
Make the stronger claim and this becomes mandatory.

**Cost.** Human time, ~1–2 h per reader for 104 items. No compute.

**Lands in.** Would move the biggest limitation into a strength.

## E8. Grounding arms, done attestably

**Question.** Does removing the evidence region cost accuracy, reported so the
repository will certify it?

`analyse_roi_4arm.py` refuses to emit a pooled four-arm figure because the arms mix
mask fills (`air` vs `local`) and the fill provenance was not recorded. That refusal
is correct. `verify_paper_tables.py` reproduces the ROI table exactly **per fill**.

To make the pooled statement available: re-run all four arms
(`full`, `roi_only`, `roi_masked`, `zero`) with the fill **recorded per item**, at
equal n, on ≥ 4 models × 4 organs. Extend coverage beyond the current 2 models.

**Outcome → claim.** Lets the grounding argument return to the main text as
"compression, not inversion" with an attestable number behind it. Without it the
claim stays a per-fill supplement table, which is survivable.

**Lands in.** Supplement, or one Discussion sentence if it comes out clean.

## E9. Input richness ablation

**Question.** Is three slices simply too little information, independent of
annotation?

Arms on the matched subset, `identified` annotation held fixed:

- 3 orthogonal slices (current)
- 9 slices (3 per axis, spanning the lesion)
- a 5×5 montage of contiguous axial slices through the lesion
- native resolution vs the current downscale

This separates "wrong slices" (E1) from "not enough slices". Higher-resolution and
more-panel inputs inflate the vision-token count sharply, so this is another
VRAM-heavy axis.

**Lands in.** Supplement.

---

# Tier 2 — optional

## E10. Complete Aria-25B-MoE

One of the 14 attempted models never finished, so the audited set is 13. The cause
was diagnosed and **already fixed**: `rhymes-ai/Aria` declares a
`vision_processor.py` it does not ship, which killed every run at load; the loader
now falls back to the native `AriaProcessor`. It is a ready-to-run 50 GB job.

```bash
python spatialgen/run_multimodel.py --model aria --qa common_subset/qa/all.jsonl \
  --out mm_aria_sighted.jsonl --device auto
python spatialgen/run_multimodel.py --model aria --qa common_subset/qa/all.jsonl \
  --blind --out mm_aria_blind.jsonl --device auto
```

It will almost certainly land at chance like the others, so it changes no
conclusion — but it removes an asymmetry a reader may notice.

## E11. Trap families across all 13 models

The scoring artefact (raw weighted silent-failure rate exactly 0.0 % because argmax
pins to one refusal string; content-free calibration puts the same models at
18.1–100 %) is demonstrated on **2 of 13 models**. That suffices for the
methodological claim the paper actually makes — *any reported refusal rate must
state its scoring rule*. Extending to 13 turns it into a model **comparison**,
which the paper does not argue. Only run this if you want that stronger claim.

## E12. Pretraining-overlap probe

MSD is public and may be in pretraining. Ask each model to identify the dataset or
reproduce a volume's metadata from the montage. Near-zero recognition strengthens
the leakage limitation; recognition would need reporting.

---

# Compute plan

Fill the cluster in this order. Jobs on different GPUs are independent — one
process per GPU, one model per process.

| wave | jobs | why now |
|---|---|---|
| **0** | E1 liver × `qwen32b` × 4 conditions, `--limit 20` | smoke test the whole path before committing hours |
| **1** | E1 full: 4 conditions × 4 organs × 4 models = 64 runs. Also E2 (minutes) and E10 (Aria, 50 GB) | E1 gates everything; E10 saturates a large card meanwhile |
| **2** | E4 large models (72B/78B/90B, `--device auto`, multi-card) + E5 self-consistency k=10 | the two genuinely VRAM- and compute-saturating axes; run once E1 tells you which condition to use |
| **3** | E3 sub-tasks, E6 native repair, E9 input richness | mechanism and robustness, all cheap per job, many jobs |
| **4** | E8 grounding re-run, E11, E12 | optional cleanup |
| **human** | E7 reader study | after E1 lands, so the input is matched |

Long jobs first on the big cards; the 8B-class models backfill smaller GPUs.

# Dependencies

```
E2 (ceiling) ──┐
               ├──> E1 (identification control) ──┬──> E4 (scale, on the winning condition)
E10 (Aria) ────┘                                  ├──> E5 (inference-time compute)
                                                  ├──> E3 (sub-task decomposition)
                                                  ├──> E9 (input richness)
                                                  └──> E7 (reader study, input-matched)
E6 (native repair) — independent
E8, E11, E12 — independent
```

**The one constraint that matters: E7 must not precede E1.** Until models are given
the same input the reader gets, a human-versus-model comparison is not a comparison
of reasoning, and running the readers first burns their time on an unusable design.

# Where results can actually go

The paper is capped at 3,000 words, 6 figures, 4 tables, and the 3D section must
stay **smaller** than the 2D section. So most of this lands in supplement or exists
to answer reviewers:

| | headline | supplement | defensive only |
|---|---|---|---|
| E1 | if it changes the result | otherwise | — |
| E2 | one sentence | one row | — |
| E4 | one table row | — | — |
| E6 | one sentence | table | — |
| E3, E5, E8, E9 | — | ✓ | — |
| E11, E12 | — | — | ✓ |
| E7 | would restructure the limitation | ✓ | — |

# Do not run

- **Re-deriving the withdrawn analyses** — contrastive amplification, the
  image-driven refusal effect, matched-pair consistency. Each was traced to a
  corpus-construction or normalisation artefact; re-running reproduces artefacts.
- **Pooled four-arm ROI without recording fill provenance.** That is precisely what
  `analyse_roi_4arm.py` refuses to certify. Either record the fill (E8) or keep the
  per-fill supplement table.
- **Quoting `figdata/` without regenerating it.** The distributed copies were stale
  for both M3D variants and disagreed with the manuscript. Run
  `make_figure_data.py` first.
- **Expanding the corpus.** 9,484 probes over 588 volumes is already far past the
  point where n limits any conclusion. More probes buy nothing; the open questions
  are all about input specification and model capability.
