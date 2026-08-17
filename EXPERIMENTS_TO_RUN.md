# Experiments to run

Ranked by whether they gate the journal submission. Experiment 1 is the only one
I consider necessary; 2 is high-value; 3–4 are optional.

Prerequisites for all of them:

```bash
export MEDVIGIL3D_ROOT=$PWD
export MSD_ROOT=/path/to/MSD          # medicaldecathlon.com, Tasks 03/06/07/10
```

You also need `cfqa_*/seg_cache/` (TotalSegmentator anatomy masks, 593 files /
544 MB). Not in this repo — regenerate with `spatialgen/run_pipeline.py` or copy
from the previous machine.

---

## 1. Identification control — **necessary before submission**

### The problem

Every published volumetric result fed the model
`montage(orthogonal_views(volume))`. In `render.orthogonal_views`, `index=None`
means **three slices through the geometric centre of the volume**, with no
annotation drawn. The question is

> "If **this lesion** grew by 5.8 mm in every direction, would it contact the heart?"

and the qid is `liver_0_lesion5_heart_g5.8` — that liver holds at least five
lesions. Nothing in the image indicates which lesion is meant, nothing marks the
target, the lesion need not appear in any of the three centre slices, and no scale
bar is drawn even though the question is metric.

So "no model's 95% CI excluded chance" has a mundane competing explanation that
says nothing about geometric reasoning: **the probe may be underspecified given
the input the model actually received.** A reviewer raises this in one sentence,
and the manuscript currently has no answer.

The reader study does not settle it. `export_reader_study.render_case` gives the
radiologist lesion-in-red, target-in-cyan, joint-visibility slices, and a 10 mm
scale bar per panel — strictly more information than any model ever got. Comparing
those two inputs is not a comparison of reasoning.

### The design

`spatialgen/run_identification_control.py` runs the same probes under a factorial
of input conditions, so any change can be attributed to a specific missing cue:

| `--condition` | slices | outlines | scale bar | isolates |
|---|---|---|---|---|
| `plain` | volume centre | no | no | the published condition (paired control) |
| `bestslice` | joint-visibility | no | no | does it need to **see** the structures? |
| `overlay` | volume centre | red / cyan | no | does it need to know **which** structures? |
| `identified` | joint-visibility | red / cyan | yes | exactly what the radiologist sees |

Panel geometry and colours match `render_case`, so `identified` is comparable with
the reader study. The prompt gains a legend sentence **only** when the
corresponding pixels are actually drawn — describing an annotation the model did
not receive would be its own confound.

### Minimum viable version

One organ, one capable model, two conditions. Liver has the most lesions per
volume, so it is where the ambiguity bites hardest:

```bash
for COND in plain identified; do
  python spatialgen/run_identification_control.py \
    --qa         cfqa_Task03_Liver/qa \
    --task-dir   $MSD_ROOT/Task03_Liver \
    --seg-cache  cfqa_Task03_Liver/seg_cache \
    --model      qwen32b \
    --condition  $COND \
    --subset     matched \
    --device     cuda:0 \
    --out        id_Task03_Liver_qwen32b_$COND.jsonl
done
```

Then add `bestslice` and `overlay` to attribute the effect, and repeat for a
second model (`internvl` or `qwen3vl` — both showed the largest image
contributions on the confounded corpus, so they are the most likely to move).

Rough cost: the matched subset is 1,368 probes; renders are cached per
(lesion, target) pair, so ~684 renders and 2 scored forward passes per probe.
Expect tens of minutes per model-condition on one GPU, not hours. Verify with
`--limit 20` first.

### Acceptance criteria

Score with the existing tooling — output format matches `mm_*`/`roi_*`:

```bash
python analysis/image_contribution_ci.py     # volume-level paired bootstrap
```

- **`identified` CI still contains 50%** → the null is about geometric reasoning
  and is now defensible: the model got everything the reader gets and still could
  not answer. Report the control in the paper; the 3D section gets materially
  stronger.
- **`identified` CI excludes 50% upward** → the published null was substantially
  an input-specification artefact. The claim must be weakened and the Results
  rewritten. Better found here than in review.
- **`bestslice` or `overlay` alone recovers most of the gain** → say which cue was
  missing. That is a finding about how volumetric probes must be presented, and it
  is publishable in its own right.

Either direction improves the paper. Run it.

---

## 2. Reader study — high value, not a blocker

`reader_study/` holds a 104-probe form (`reader_form.csv`) and the answer key
(`ANSWER_KEY.csv`). **Every `reader_answer` cell is empty — no radiologist has
completed it.** Regenerate the images with `spatialgen/export_reader_study.py`.

Why it is worth doing: the form already carries a
`clinical_relevance__1to5` column and `severity_agree__yes_no`. Filling it would
convert the paper's largest limitation — "the volumetric reference standard is
geometric and was not confirmed clinically salient" — into a positive result, and
it answers the "why should a radiologist care about lesion-growth-contact probes?"
question directly.

Why it is not a blocker: the external review confirmed this explicitly,
**conditional on** the volumetric section never claiming a clinician-validated
reference standard. The manuscript currently calls that section *external testing*
and states plainly that no reader confirmation was performed. Keep that wording
and the paper stands without this. Make the stronger claim and the study becomes
mandatory.

Cost: human time, roughly 1–2 hours per reader for 104 items. No compute.

**Do experiment 1 first.** Until the models are given the same input the reader
gets, a reader-versus-model number is not interpretable.

---

## 3. Complete Aria-25B-MoE — cosmetic

One model of the 14 attempted never finished, so the audited set is 13. Finishing
it removes an asymmetry a reader may notice. It will almost certainly land at
chance like the others, so it changes no conclusion.

```bash
python spatialgen/run_multimodel.py --model aria --qa common_subset/qa/all.jsonl \
  --out mm_aria_sighted.jsonl --device cuda:0
python spatialgen/run_multimodel.py --model aria --qa common_subset/qa/all.jsonl \
  --blind --out mm_aria_blind.jsonl --device cuda:0
```

---

## 4. Extend the trap families to all 13 models — optional

The silent-failure scoring artefact (raw weighted rate is exactly 0.0% because
argmax pins to one refusal string; content-free calibration puts the same models
at 18.1–100%) is currently demonstrated on **2 of 13 models**.

That is sufficient for the claim the manuscript actually makes, which is
methodological: *any reported refusal rate must state its scoring rule.* Extending
to 13 models would turn it into a model **comparison**, which is not what the
paper argues. Only do this if you want to make that stronger claim.

---

## Not worth running

- **Re-running the four-arm ROI arms to recover mask-fill provenance.** The pooled
  four-arm number is not defensible and has been demoted to a supplement table
  reported per fill (`air` vs `local`), where `verify_paper_tables.py` reproduces
  it exactly. `analyse_roi_4arm.py` refusing to emit a pooled figure is the guard
  working correctly, not a bug to fix.
- **The three withdrawn analyses** (contrastive amplification, the image-driven
  refusal effect, matched-pair consistency). Each was traced to a
  corpus-construction or normalisation artefact. They are documented as withdrawn;
  re-running them would reproduce artefacts.
