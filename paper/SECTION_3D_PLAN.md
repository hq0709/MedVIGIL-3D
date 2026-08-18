# The 3D section, organised around what the runs actually support

Budget from EXPERIMENTS_TO_RUN.md: 3,000 words, 6 figures, 4 tables total, and
the 3D section must stay **smaller** than the 2D one. So this plans for roughly
**900 words, 2 figures, 1 table** in the main text, with everything else in the
supplement or held for review.

Every number below is from `results_new/` on the corrected renderer. Numbers
produced before the render fix are quarantined in `results_broken_render/` and
must not be quoted.

---

## 1. What changed about the claim

The old sentence was *"no model's CI excluded chance, so volumetric VLMs cannot
do geometric reasoning."* Three results make that both weaker and less
interesting than what the data actually shows.

**The arithmetic is intact.** Asked the bare comparison the task reduces to —
"is 21.4 greater than or equal to 18.16?" — every model scores **100.0%**
(7B, 8B, 8B, 32B, 72B), balanced predictions, zero pair-identical answers.

**The failure reproduces with no image at all.** Hand the model the same two
numbers inside the clinical sentence and remove the image entirely: **50.0–52.4%**,
with a single constant answer (Qwen2.5-VL-7B answers "no" 1368/1368; InternVL3
and Qwen3-VL answer "yes" 1368/1368). Greedy decoding gives the identical
constant, so it is not a scoring artefact.

**The image channel is not dead — the metric part of it is.** Sub-task
decomposition on the annotated rendering (§3) shows perception and naming work
and metric estimation does not.

So the claim to make is a **located** failure, not a global one:

> Models perceive the annotation, partially identify the structures, and execute
> the comparison perfectly when it is stated numerically. What they cannot do is
> recover a metric quantity from the volumetric input. A second, independent
> failure sits downstream: even when the distance is supplied in text and no
> image is involved, the clinical phrasing collapses every model to a constant
> answer.

This is stronger than the original because it pre-empts the obvious objection —
*"are you sure you are measuring vision?"* — by answering it inside the paper.

---

## 2. Main text, paragraph by paragraph

### ¶1 Setup (~120 words)
Counterfactual probes with computed ground truth: 9,484 probes, 588 CT volumes,
4 MSD organs. State the reference rule and that it verifies: the
`geometry-oracle` arm, which applies *contact iff gap ≤ growth* to the stored
provenance, reproduces the labels on **8,476/8,476 growth-contact probes
(100.0%)**. This is the sentence that licenses everything after it.

### ¶2 The null, and that identification does not lift it (~200 words) → **Table 1**
Motivate with the audit fact, which is quantitative and damaging if a reviewer
finds it first: in the published condition the lesion the question names has
**no voxel on any slice shown for 42.0%** of probes, and both structures appear
in only 54.2%. The identification control repairs exactly this (100% visibility
under `bestslice`/`identified`) and changes nothing:

| model | plain | bestslice | overlay | identified |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 50.0 | 50.0 | 50.0 | 50.0 |
| InternVL3-8B | 50.0 | 50.3 | 49.6 | 50.4 |
| Qwen3-VL-8B | 48.2 | 49.9 | 49.5 | **52.4** |
| Qwen2.5-VL-32B | 50.7 | 51.2 | 51.0 | **52.6** |
| Qwen2.5-VL-72B | 49.2 | — | — | 50.0 |

Two of twenty intervals exclude 50 and both by under 3 points. Report the
**matched-pair endpoint beside the pooled one** (§5) — do not report only one.

Add the scale sentence here rather than as its own paragraph: Qwen2.5-VL-72B,
146.8 GB of weights sharded over two H100s, is at **49.2% / 50.0%**. The null
does not close over a 10× parameter range.

The detail worth one clause: identification does not merely fail to help, it
makes the models **more** degenerate. Pair-identical answers rise monotonically
with information given — InternVL3 81→91%, Qwen2.5-VL-72B **89→99%**, whose
yes-rate reaches 95.0% under `identified`.

### ¶3 Where the failure lives (~250 words) → **Figure 1**
The core result. Same annotated image, same four-option format, only the
question changes:

| model | which organ holds the red outline | same, annotation removed | which structure is cyan | **distance in mm** |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 67.7 | 57.3 | 34.3 | **25.7** |
| InternVL3-8B | 41.0 | 15.0 | 31.3 | **26.3** |
| Qwen3-VL-8B | 81.7 | 49.7 | 60.0 | **23.0** |
| Qwen2.5-VL-32B | 76.0 | 52.0 | 45.0 | **28.3** |

Chance is 25%. Three things to say:

1. **The annotation is perceived.** Localisation runs well above chance, and
   removing the outline costs 10–32 points — the no-annotation control is what
   turns "it answered" into "it used the overlay".
2. **Metric estimation is absent.** With both structures outlined and a 10 mm
   scale bar drawn in the panel, all four models sit at chance, and their
   answers collapse onto one or two buckets (Qwen2.5-VL-32B answers "15" on
   298/300 items) while the gold buckets are near-uniform (85/67/79/69).
3. **It is not a dead channel.** The same model on the same image answers a
   categorical question at 76% and the metric question at 28% with a constant
   response. Categorical information survives the pipeline; metric information
   does not.

### ¶4 The second failure, and the ceiling (~150 words)
`numeric-oracle` 100% vs `text-oracle` 50% with a constant answer, no image in
either. State that this bounds the task from above (it *is* solvable) and that
it locates a failure that has nothing to do with vision. One sentence on
inference-time compute is owed here and is **not yet run** (§6).

### ¶5 Limitations (~180 words)
Three that a reviewer will otherwise find:

- **Not thirteen independent results.** On the growth-matched subset, **7 of 13
  models put ≥96% of answers on one token**, and on a balanced corpus a modal
  share of *m* confines accuracy to [m−0.5, 1.5−m] whatever the model is doing:
  InternVL3-14B's 50.1% is pinned inside [46.5, 53.5]. After also excluding the
  three native models that fail the response-channel controls, **4 models** carry
  the claim. Report the modal-share column; do not say "13 models agree".
- **The reporting unit carries a cue.** Matching on growth removed the growth
  cue (69.2%→50.8% single-threshold ceiling) and created a gap cue
  (50.0%→**72.0%**). Complete pairs are gap-neutral but growth-informative
  (83.7%). Neither unit is clean; say which endpoint each number is on.
- **The clinically nearest cases are excluded by construction.** `margin=2.0`
  removes every probe with gap ≤ 2 mm — measured: 0 of 2,262, observed range
  2.04–40.00 mm. The lesions already abutting a critical structure, where
  management actually turns, are not in the corpus.

---

## 3. Figures and tables

**Figure 1 (main).** Sub-task decomposition: four grouped bars per model
(localise / localise-without-annotation / name / distance) against a 25% chance
line. This is the paper's mechanism in one picture.

**Figure 2 (main).** Identification control: 5 models × 4 conditions, pooled
accuracy with volume-clustered 95% CIs, chance line at 50, with a second panel
showing pair-identical rate rising with information given. The second panel is
the counter-intuitive result and deserves the space.

**Table 1 (main).** The E1/E4 grid above, with the 72B row.

**Supplement.** Input audit per organ; the ceiling arms in full
(`geometry-oracle`, `numeric-oracle`, `text-oracle`, `text-oracle-blind`, and
the `-gen` decoding controls); the confound table; the modal-share table; input
richness (§6); native-model response channels.

**Defensive only, not in the paper.** The render-parity work (`plain` is
bit-exact against `montage(orthogonal_views(...))`, `identified` byte-identical
to the reader-study image, `run_identification_control.py selftest`). Keep it in
the repo; it answers "is your control arm really the published condition?" in
one command if asked.

---

## 4. Sentences in the current draft that must change

| now | should be |
|---|---|
| "13 models, none above chance" | "4 models with verified, non-degenerate response channels; the other 9 are listed with the reason each is uninformative" |
| "models cannot perform volumetric geometric reasoning" | "models do not recover metric geometry from volumetric input; annotation perception and the comparison itself are intact" |
| accuracy reported on the matched subset only | pooled **and** matched-pair accuracy, with the cue each unit controls stated |
| "chance-level accuracy means no information was extracted" | drop — a gap threshold reaches 72.0% on this subset |
| the 41-point framing cost for M3D-LaMed-Phi3 | do not quote until §6 is resolved |

---

## 5. What is finished

| | status |
|---|---|
| E1 identification control | **64/64**, 4 models × 4 conditions × 4 organs, corrected renderer |
| E4 scale | **8/8** (Qwen2.5-VL-72B, plain + identified × 4 organs) |
| E2 ceiling + decoding/framing controls | **24 arms** across 5 models |
| E3 sub-task decomposition | **16/16**, including the no-annotation control |
| Input audit | 1,368 probes, all four conditions |
| Corpus verification | geometry-oracle 100% on 8,476 probes |
| Reader study materials | **104/104** images on the corrected renderer, 52 yes / 52 no |

Zero job failures across the final campaign.

---

## 6. Three gaps, ranked

1. **The metric result needs its "three slices is too few" control.** E3 shows
   models cannot read a distance off three annotated planes. The obvious
   rebuttal is that three planes do not contain it. E9 partly answers this on
   the *composite* question — InternVL3 with 25 contiguous axial slices at
   6.72 MPx (5.7× the pixels) scores **52.2%** with 96% pair-identical answers,
   and the magnified `zoom` arm 58.8% — but the composite is masked by the
   downstream failure, so it is supporting rather than decisive. **The clean
   experiment is the distance sub-task asked under `axial25` and `zoom`**, about
   twenty minutes of compute. Missing: `qwen32b × axial25` (4 runs, skipped by a
   VRAM estimate set 2.4 GB too high; corrected in `runs/make_program.py`).
2. **Inference-time compute (E5) is not run.** The 2D half ships CoT and
   self-verification ablations; a 3D section without them invites the question
   for free. Code is written (`spatialgen/run_inference_compute.py`), job list
   ready (`runs/jobs_e5.jsonl`).
3. **The native-model numbers do not currently reproduce.** Re-running
   `sanity_controls.py` against the committed `sanity_m3d.jsonl` gives **42.9%
   against the committed 76.4%** on the same 20 volumes, agreeing on 63.6% of
   items — same script, different library stack. Until a same-stack campaign is
   run (`runs/jobs_e6.jsonl`, 6 jobs), neither the 29.3-point nor the 41-point
   framing effect is quotable.

Optional and deliberately not run: E8 four-arm grounding (keep the per-fill
supplement table; `analyse_roi_4arm.py` refusing is correct behaviour), E10 Aria,
E11 trap families across 13 models (Tier 2, and the paper does not make the
comparison claim it would support).

One experiment worth adding that is not in the original programme: a **metric
floor test** — two markers a known distance apart on a synthetic image with the
same scale bar. If models are at chance there too, the finding generalises
beyond CT and beyond medicine; if they are not, it localises to medical imaging.
Either outcome is stronger than the present scope, and it costs minutes.
