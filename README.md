# MedVIGIL-3D — volumetric evaluation harness

> ### ➤ Start here
> - **[paper/SECTION_3D.md](paper/SECTION_3D.md)** — the volumetric result as it
>   now stands, with every number recomputed from `results_new/`.
> - **[EXPERIMENTS_REMAINING.md](EXPERIMENTS_REMAINING.md)** — what is left, why
>   it matters, and the exact command. Two items block submission.
> - [EXPERIMENTS_TO_RUN.md](EXPERIMENTS_TO_RUN.md) — the original programme,
>   kept for provenance. E1–E4 and E6 are done; read it alongside the two above.

Annotation-free counterfactual probing of medical vision–language models on CT.
Probe answers are **computed** from geometry rather than annotated: given an
expert lesion mask and an automatic anatomy segmentation, *"if this lesion grew
by d mm in every direction, would it contact the aorta?"* is answered exactly by
one distance transform — contact occurs iff the lesion-to-target surface gap ≤ d.

9,484 probes / 588 CT volumes / 4 Medical Segmentation Decathlon organs / 13 models.

---

## The experiment to run first

`spatialgen/run_identification_control.py` — **the identification control.**

Every published volumetric result was produced by feeding the model
`montage(orthogonal_views(volume))`. Read `render.orthogonal_views`: with
`index=None` it takes three slices through the **geometric centre of the volume**
and draws no annotation. But the question is *"if **this lesion** grew by 5.8 mm
…"* and the qid is `liver_0_lesion5_heart_g5.8` — there are at least five lesions
in that liver. Nothing in the image says which one, nothing marks the target, the
lesion need not intersect any of the three centre slices, and no scale bar is
drawn although the question is metric.

So the headline null ("no model's 95% CI excluded chance") currently admits a
mundane reading unrelated to geometric reasoning: **the probe may be
underspecified given the input the model received.** The reader study does not
settle it — `export_reader_study.render_case` gives the radiologist lesion-in-red,
target-in-cyan, joint-visibility slices and a 10 mm scale bar, i.e. strictly more
information than any model got.

The control runs the same probes under a factorial of input conditions so any
change can be attributed:

| `--condition` | slices | annotation | scale bar | question it answers |
|---|---|---|---|---|
| `plain` | volume centre | none | no | reproduces the published condition |
| `bestslice` | joint-visibility | none | no | does it just need to *see* the structures? |
| `overlay` | volume centre | red/cyan outlines | no | does it just need to know *which* structures? |
| `identified` | joint-visibility | red/cyan outlines | yes | exactly what the radiologist sees |

```bash
export MEDVIGIL3D_ROOT=$PWD
export MSD_ROOT=/path/to/MSD

python spatialgen/run_identification_control.py \
  --qa         cfqa_Task03_Liver/qa \
  --task-dir   $MSD_ROOT/Task03_Liver \
  --seg-cache  cfqa_Task03_Liver/seg_cache \
  --model      qwen32b \
  --condition  identified \
  --subset     matched \
  --out        id_Task03_Liver_qwen32b_identified.jsonl
```

Run `plain` too — it is the paired control, and it should reproduce the published
number. Output format matches the existing `mm_*` / `roi_*` runs, so the analysis
scripts read it unchanged.

**Interpreting it.** If `identified` stays at chance, the null is about geometric
reasoning and is defensible: the model got everything the reader got and still
could not answer. If `identified` clears chance, the published null was
substantially an input-specification artefact and the claim must be weakened.
Both outcomes are worth having; the second is much better found here than in
review.

---

## Data you must supply

Two things are **not** in this repo:

| What | Why | How to get it |
|---|---|---|
| **MSD volumes** (`imagesTr/`, `labelsTr/`) | licensed upstream, ~100 GB | http://medicaldecathlon.com — Tasks 03, 06, 07, 10 |
| **`cfqa_*/seg_cache/`** — TotalSegmentator anatomy masks, 593 files / 544 MB | too large to clone comfortably | regenerate with `spatialgen/run_pipeline.py`, or copy from an existing run |

`reader_study/images/` (104 rendered PNGs) is also excluded; regenerate with
`spatialgen/export_reader_study.py`.

Everything else needed to reproduce the published tables **is** here: probe
definitions (`cfqa_*/qa/`), the 243 result files from the completed runs, the
frozen figure data, and the analysis scripts.

Paths resolve from the repo root. Override with `MEDVIGIL3D_ROOT`; point
`MSD_ROOT` at the decathlon data.

---

## Reproducing the published numbers

```bash
python growth_matched.py        # the growth-matched subset + 13-model table
python verify_paper_tables.py   # asserts the headline and ROI tables
python analyse_models.py        # cross-model summary on the 2,262 common subset
python analyse_traps.py         # calibrated silent-failure rates
python analyse_margin.py        # logit perturbation, decision gap, flip rate
python make_figure_data.py      # regenerates figdata/ from raw
```

`growth_matched.py` and `verify_paper_tables.py` run without the segmentation
cache (they re-score stored predictions). The rest of the pipeline needs the
volumes.

⚠️ **Regenerate `figdata/` before quoting it.** The copies distributed with an
earlier snapshot were stale for two models (both M3D variants) and disagreed with
the manuscript; `make_figure_data.py` rebuilds them from stored predictions.

⚠️ `analyse_roi_4arm.py` deliberately **refuses** to emit a pooled four-arm figure
unless all four arms are present at equal length with recorded mask-fill
provenance. It currently refuses. That is the guard working: `air` and `local`
fills are not interchangeable, and `verify_paper_tables.py` reproduces the ROI
table exactly when the arms are split per fill.

---

## Layout

```
spatialgen/                     corpus generation, rendering, inference
  run_identification_control.py the control described above, plus the ceiling
                                arms; `selftest` asserts render parity
  run_subtasks.py               sub-task decomposition (perceive/name/measure)
  run_input_richness.py         how much volume is shown, annotation held fixed
  run_inference_compute.py      chain-of-thought, self-consistency, verification
  run_leakage.py                pretraining-overlap probes with a positive control
  make_seg_cache.py             regenerates cfqa_*/seg_cache (lock-coordinated)
  verify_seg_provenance.py      checks a regenerated cache against stored gap_mm
runs/                           orchestration; job lists are regenerated, not stored
  run_queue.py                  VRAM-aware queue, atomic claims, share-aware budget
  make_jobs.py, make_program.py generate the job lists
  prerender.py                  fills the render cache on CPU
  audit_conditions.py           what each condition actually shows (pixels, no model)
  summarise.py                  every table in one command
  director.sh, e4_watcher.sh    wave scheduling; the 72B takes both cards when free
results_new/                    this campaign's runs + FINAL_{SUMMARY,CI,CLAIMS}.txt
paper/SECTION_3D.md             authoritative volumetric section
paper/examples/                 example renderings of the input-richness arms
  run_identification_control.py the control described above
  run_cfqa.py                   counterfactual probe generation
  run_multimodel.py             multi-model montage inference
  run_roi_arms.py               four-arm ROI ablation
  export_reader_study.py        reader-study export (overlay renderer)
  render.py, scene_graph.py     windowing, orthogonal views, RAS loading
  lesion_binding.py             lesion instance binding
growth_matched.py               growth-confound removal + matched subset
verify_paper_tables.py          asserts manuscript tables against raw
analyse_*.py                    per-analysis reporting
analysis/                       primary endpoint, image-contribution CIs, figures
cfqa_Task*/qa/                  probe definitions (+ seg_cache, excluded)
common_subset/                  the 2,262-probe subset every model was scored on
figdata/                        frozen figure data
*.jsonl                         243 result files from completed runs
paper/PAPER.md                  internal source of record — see note below
reader_study/                   104-probe reader form + answer key
```

`paper/PAPER.md` is a **superseded internal draft**, retained only because
`verify_paper_tables.py` asserts against it. It is not the submission, and its
contribution list quotes two accuracy ranges that the raw data does not support
(the abstract's ranges are the correct ones). Do not cite it.

---

## Known state

- The volumetric claim is now **located** rather than global: models perceive the
  annotation and execute the comparison exactly, and cannot estimate distance
  from the image. See `paper/SECTION_3D.md`. §7 of `paper/PAPER.md` predates this.
- Seven of the thirteen models place ≥96 % of answers on one token; on a balanced
  corpus that confines accuracy near 50 % regardless of reasoning. Four models
  carry the claim. Do not report the thirteen as agreeing.
- The growth-matched subset removes the growth cue and introduces a gap cue
  (single-threshold ceiling 72.0 % against 50.0 % on the full corpus). Report the
  endpoint each number is on.
- The committed native-model response-channel files do not reproduce on the
  current library stack (42.9 % against a recorded 76.4 %). See
  EXPERIMENTS_REMAINING.md R2.
- One model (Aria-25B-MoE) did not complete; the audited set is 13 of 14 attempted.
- Coverage is a thin diagonal: 13 models on the counterfactual corpus, 4 on the
  ROI arms, 4 on target pairs, 2 on the trap families.
- `tp_qwen3vl_blind.jsonl` is a truncated run (1,144 of 1,200 rows) and
  `tp_internvl14_sighted.jsonl` is a 48-row fragment; both are excluded from
  reported results.
- Three analyses were withdrawn during the original audit after controls traced
  their effects to corpus-construction or normalisation artefacts.

## Licence and data provenance

Code released for review and replication. The probe definitions are derived from
the Medical Segmentation Decathlon (CC BY-SA 4.0); MSD terms govern the imaging
data, which is not redistributed here. No credentialed or protected health
information is included in this repository.
