# Audit II — Volumetric: what fails, and where

*Authoritative text for the 3D section. Every number is recomputed from
`results_new/` by `runs/summarise.py`, `analysis/identification_control_ci.py`
and `analysis/verify_claims.py`. Supersedes §7 of `paper/PAPER.md`, which
predates the identification control and the sub-task decomposition.*

---

## 1. What this audit establishes

Volumetric probe answers are **computed** from a deterministic simulator rather
than annotated: given an expert lesion mask and an automatic anatomy
segmentation, *if this lesion grew d mm in every direction, would it contact the
aorta?* is decided exactly by whether the lesion-to-target surface gap is at most
d. The corpus is 9,484 probes over 588 CT volumes and four Medical Segmentation
Decathlon organs, at zero annotation cost.

The reference rule reproduces the stored labels on **8,476 of 8,476**
growth-contact probes (100.0%). Every accuracy below is measured against labels
that regenerate from their own recorded provenance.

The headline is not that models fail. It is **where** they fail. Decomposing the
probe into its four constituent steps, on the same images and the same scoring
rule, gives:

| step | what it asks | result |
|---|---|---|
| perceive | which organ holds the red-outlined lesion | 41.0–81.7% (chance 25%) |
| identify | which structure is outlined in cyan | 31.3–60.0% |
| **measure** | **how many mm separate the two outlined structures** | **23.0–28.3% (chance 25%)** |
| compare | is 21.4 ≥ 18.16 | **100.0%** |

Perception works, the comparison is exact, and the metric step is absent. A
second, independent failure sits downstream of all four: supplying the distance
in text and removing the image entirely still collapses every model to a single
constant answer.

---

## 2. The published input was underspecified, and repairing it changes nothing

Every previously reported volumetric result fed the model
`montage(orthogonal_views(volume))`, which takes three slices through the
geometric centre of the volume and draws no annotation, while the question asks
about "this lesion" in a scan that may hold five. Measured over the 1,368-probe
growth-matched subset, in that condition **the lesion the question names has no
voxel on any slice shown for 42.0% of probes**, and both structures appear
together in only 54.2%.

The identification control varies the input over four conditions, holding the
probe fixed: `plain` (the published condition, reproduced bit-exact),
`bestslice` (joint-visibility slices), `overlay` (centre slices, lesion in red
and target in cyan), and `identified` (joint-visibility slices, both outlines,
and a 10 mm scale bar — exactly what the reader study shows a radiologist).
`bestslice` and `identified` put both structures on screen for 100% of probes.

Pooled accuracy, volume-clustered 95% CIs, B=10,000:

| model | plain | bestslice | overlay | identified |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 50.0 [48.2, 51.7] | 50.0 | 50.0 | 50.0 |
| InternVL3-8B | 50.0 [47.8, 52.3] | 50.3 | 49.6 | 50.4 [48.4, 52.4] |
| Qwen3-VL-8B | 48.2 [46.3, 50.3] | 49.9 | 49.5 | **52.4 [50.3, 54.6]** |
| Qwen2.5-VL-32B | 50.7 [48.7, 52.6] | 51.2 | 51.0 | **52.6 [50.7, 54.6]** |
| Qwen2.5-VL-72B | 49.2 [47.2, 51.2] | — | — | 50.0 [48.1, 51.9] |

Twenty cells span 48.2–52.6%. Two intervals exclude 50, both by under three
points, and with twelve paired contrasts tested on two endpoints a nominal 95%
interval will exclude chance about once by construction; we do not read either as
a capability. Giving a model everything the radiologist is given does not make
this probe answerable.

**Scale does not close it.** Qwen2.5-VL-72B — 146.8 GB of bf16 weights sharded
across two H100s — is at 49.2% and 50.0%. Over a 10× parameter range, from 7B to
72B, nothing changes. (InternVL3-78B at 156.8 GB and Llama-3.2-90B-Vision at
177.2 GB exceed the 159 GB available on this pair of cards and were not run.)

**Information makes the models more degenerate, not less.** The share of matched
pairs answered identically — geometrically impossible, since the two members
straddle the true gap — rises with the information supplied for three of the four
non-degenerate models: InternVL3-8B 81→91%, Qwen2.5-VL-32B 90→92%, and
Qwen2.5-VL-72B **89→99%**, whose yes-rate reaches 95.0% under `identified`.
Qwen3-VL-8B moves the other way (94→85%). Qwen2.5-VL-7B answers "no" to all
1,368 probes in all four conditions: identification does not unstick a degenerate
responder.

On the matched-pair endpoint the identification effect is small and
model-dependent in sign: −5.0 [−7.4, −2.5] for InternVL3-8B, −3.9 [−6.1, −1.6]
for Qwen2.5-VL-32B, −5.2 [−7.0, −3.6] for Qwen2.5-VL-72B, and +4.5 [+2.5, +6.7]
for Qwen3-VL-8B. No model reaches a level that would indicate the probe became
answerable.

---

## 3. Where the failure lives

The composite probe requires four things at once. Asked separately, on the
`identified` rendering, in the same four-option format and under the same
likelihood scoring (300 probes per cell, stratified over the four organs, chance
25%):

| model | localise (organ of the red outline) | same, annotation removed | name the cyan structure | **distance in mm** |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 67.7 | 57.3 | 34.3 | **25.7** |
| InternVL3-8B | 41.0 | 15.0 | 31.3 | **26.3** |
| Qwen3-VL-8B | 81.7 | 49.7 | 60.0 | **23.0** |
| Qwen2.5-VL-32B | 76.0 | 52.0 | 45.0 | **28.3** |

Three things follow.

**The annotation is perceived and used.** Localisation runs well above chance,
and the no-annotation control — the same question on an unannotated rendering,
with the legend sentence dropped — costs 10.4 to 32.0 points. Without that
control, high localisation accuracy would be consistent with recognising the body
region and never seeing the outline at all.

**Metric estimation is absent.** With both structures outlined and a 10 mm bar
drawn in the panel from that panel's own post-resampling spacing, all four models
sit at chance. Their answers collapse onto one or two of the four buckets —
Qwen2.5-VL-32B answers "15" on 298 of 300 items — while the correct buckets are
near-uniform (85/67/79/69).

**This is not a dead output channel.** The same model, on the same image, in the
same four-option format, answers a categorical question at 76.0% and the metric
question at 28.3% with a near-constant response. Categorical information survives
the pipeline; metric information does not.

The distance options are 5/15/25/35 mm rather than the 5/15/30/60 one might
choose by eye: the corpus caps the gap at 40 mm, so an option at 60 is never
correct, and a never-correct option is a hint rather than a distractor. The
bucket edges (10, 20, 30) sit on the observed quartiles (10.4, 19.7, 29.4), so
all four are close to equiprobable.

---

## 4. A second failure, with no image in it

Two arms bound the task from above. `geometry-oracle` applies the reference rule
to the stored provenance and scores 100.0%, verifying the harness rather than a
model. `numeric-oracle` states the comparison the task reduces to, stripped of
clinical wording — *is 21.4 greater than or equal to 18.16?* — and every model
scores **100.0%**, with balanced predictions and no pair answered identically.

`text-oracle` supplies the same two numbers inside the clinical sentence and
removes the image entirely. Every model collapses:

| model | numeric-oracle | text-oracle | modal answer under text-oracle |
|---|---|---|---|
| Qwen2.5-VL-7B | 100.0 | 50.0 | "no", 1368/1368 |
| InternVL3-8B | 100.0 | 50.0 | "yes", 1368/1368 |
| Qwen3-VL-8B | 100.0 | 50.0 | "yes", 1368/1368 |
| Qwen2.5-VL-32B | 100.0 | 51.1 | "no", 1353/1368 |
| Qwen2.5-VL-72B | 100.0 | 52.4 | "yes", 1335/1368 |

Greedy decoding with answer parsing gives the identical constants, so this is not
an artefact of scoring by likelihood over option strings. The task is solvable —
we have shown the ceiling — and the collapse reproduces with no image present.
Any account of the volumetric null must therefore include a component that is not
visual.

---

## 5. Is three slices simply too little?

The remaining mundane explanation for a null visual channel is that three planes
out of several hundred do not contain the geometry. Holding the annotation at the
`identified` level and varying only how much of the volume is shown (80 probes
per organ, pair-preserving):

| arm | input | InternVL3-8B | Qwen2.5-VL-32B |
|---|---|---|---|
| slices3 (three orthogonal planes) | 1.18 MPx | 54.1 | 50.9 |
| slices9 (three per axis, spanning the lesion) | 2.39 MPx | 50.6 | 52.2 |
| axial25 (5×5 contiguous axial slices) | 6.72 MPx | 52.2 | *not run* |
| zoom (same slices, cropped to the structures and magnified back) | 1.14 MPx | 58.8 | 44.4 |

Multiplying the visible volume by 5.7× leaves InternVL3-8B at 52.2% with 96% of
pairs answered identically. This is supporting rather than decisive evidence,
and we say so: these arms ask the *composite* question, which §4 shows collapses
even when the distance is supplied in text, so the downstream failure masks
whatever the metric channel is doing. **The clean version of this control — the
distance sub-task of §3 asked under `axial25` and `zoom` — is not yet run.**

---

## 6. What these numbers do not license

**Not thirteen independent results.** On the growth-matched subset, seven of the
thirteen audited models place at least 96% of their answers on one token. On a
balanced corpus a modal share of *m* confines accuracy to [*m*−0.5, 1.5−*m*]
whatever the model is doing: InternVL3-14B's 50.1% is pinned inside [46.5, 53.5]
by its answer distribution alone. Excluding those seven, and the three native
volumetric models that fail the known-answer response-channel control, **four
models carry the claim**: Qwen3-VL-8B (48.7), InternVL3-8B (50.1), Pixtral-12B
(51.0) and Qwen2.5-VL-32B (50.6). All four intervals still contain 50. We report
the modal-share column alongside every accuracy and do not describe the thirteen
as agreeing.

**Chance is not the same as "no information extracted".** The growth-matched
subset was constructed to remove the growth-magnitude cue, and it does: a single
threshold on the growth amount falls from 69.2% on the full corpus to 50.8%.
Inside a narrow growth bin, however, "yes" means gap ≤ growth, so the yes-items
are exactly the small-gap ones, and a single threshold on the **gap** rises from
50.0% to **72.0%**. Complete matched pairs are gap-neutral (50.0%) but
growth-informative (83.7%). Neither unit is free of both numbers, so every figure
is reported on a stated endpoint, and no model comes near either ceiling.

**The clinically nearest cases are absent by construction.** Probe generation
requires the gap to exceed a 2 mm margin, which removes every case where the
lesion already abuts the target: the emitted corpus contains 0 probes with gap
≤ 2 mm, observed range 2.04–40.00 mm. Lesions in contact with a critical
structure — where management actually turns — are outside this audit.

**The native volumetric models remain unmeasurable, and their published
control numbers do not currently reproduce.** Re-running the response-channel
controls today against the committed file gives 42.9% where 76.4% is recorded,
on the same 20 volumes and agreeing on 63.6% of items — the same script under a
different library stack. Until a single-stack campaign is run, we quote no
framing-repair effect for these models.

---

## 7. Reproducibility of the input pipeline

The control arm must be the published condition or the comparison is vacuous, and
this is a property of pixels rather than of accuracies: two renderings that
differ can still both score 50%. `run_identification_control.py selftest` asserts
on a synthetic anisotropic volume that `plain` is bit-exact against
`montage(orthogonal_views(...))`, that all four conditions emit identically
shaped input, that `identified` is byte-identical to the reader-study rendering,
and that the annotated arms agree with `plain` outside their annotation. The
reader-study export and the model input are generated through one shared
orientation and compositing rule, so a reader-versus-model comparison is a
comparison of reasoning rather than of image geometry.

---

## 8. Space in the manuscript

The paper allows 3,000 words, 6 figures and 4 tables in total, and the 3D
section must stay smaller than the 2D one. This text is the authoritative
source, not the submission draft; compressed to roughly 900 words it becomes:

**Figure 1 (main).** Sub-task decomposition — four grouped bars per model
(localise / localise-without-annotation / name / distance) against a 25% chance
line. The mechanism in one picture.

**Figure 2 (main).** Identification control — five models × four conditions,
pooled accuracy with volume-clustered intervals against a 50% line, and a second
panel showing pair-identical rate rising with the information supplied.

**Table 1 (main).** The condition grid of §2 including the 72B row.

**Supplement.** Input audit per organ; the ceiling arms in full, including the
`-gen` decoding controls; the confound table of §6; the modal-share table;
input richness (§5); the native-model response channels.

**Held for review, not printed.** The render-parity work of §7. It answers "is
your control arm really the published condition?" in one command
(`run_identification_control.py selftest`) if a reviewer asks, and costs nothing
until then.
