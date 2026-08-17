# MedVIGIL: Evidence-Conditional Trustworthiness of Medical Vision–Language Models across 2D and Volumetric Imaging

*Journal extension of the conference version. The framework, probe families,
metric definitions and composite score are unchanged; this version adds a second
modality (3D CT), a simulator-backed construction of the visual-counterfactual
arm, and a cross-modality analysis that the 2D suite alone could not support.*

---

## Abstract

Medical vision–language models are audited almost entirely in 2D, where every
probe must be authored and adjudicated by clinicians. That coupling caps audits
at a few hundred items and leaves the volumetric regime — where most diagnostic
CT and MR work happens — largely unexamined. We introduce **verifiable
counterfactual probing**, in which probe answers are *computed* from a
deterministic anatomical simulator rather than annotated: given an expert lesion
mask and an automatic anatomy segmentation, we ask about states that do not
exist in the scan — *if this lesion grew 12 mm, would it contact the aorta?* —
whose answers follow exactly from geometry. Because the queried state never
occurred, no report or textbook describes it; because the answer is computed, it
is verifiable to the voxel and the corpus is bounded only by compute.

We build 9,484 probes over 588 CT volumes spanning four organs at zero
annotation cost, and audit thirteen models across eight families, both
general-purpose models fed orthogonal-view montages and native volumetric
medical VLMs. **On a growth-matched subset where the question's numeric
parameter carries no label information, every model sits at the 50 % chance
level (48.7–51.0 %, all thirteen intervals containing 50) with image
contributions from −1.3 to +1.0 pp.** Four of the thirteen cannot be scored at
all: a known-answer response-channel control shows their output is pinned to one
answer regardless of input, and the four are not predictable from architecture —
three are native volumetric systems and one is a general-purpose model that
answers "no" to 98.6 % of everything, including whether the image is a CT.

Two findings concern the instruments rather than the models. First, MedVIGIL's
Visual Grounding Ratio inverts in 3D, and a four-arm decomposition says why:
removing the evidence region costs nothing (−3.5 to +1.8 pp across thirteen
organ-model cells in four families) while the volume as a whole is worth
3.3–10.4 points. VGR subtracts an arm that replaces 99 % of the volume from one
that replaces 1 %, so it cannot separate grounding from intactness — a
distinction 2D never forced. Second, likelihood scoring over heterogeneous
option strings reports a weighted silent-failure rate of exactly 0.0 % for every
model on every trap family, because a refusal string dominates regardless of
content; content-free calibration raises the same models to 43.7–100.0 %.

We report three results we withdrew during the audit, and the checks that caught
them, because each survived every guard conventional in this literature: a
capability gain from contrastive amplification that came from a text-side cue in
our own corpus, an image-driven refusal effect that came from normalising two
arms against different baselines, and a matched-pair consistency measure that
was equivalent to comparing two integers in the prompt. The volumetric visual
pathway is not inert — amplifying the sighted-versus-blind logit difference cuts
one model's weighted silent-failure rate by 17.4 pp — but on these probes none
of the signal it carries is located in the lesion, the target, or the corridor
between them.

---

## 1. Introduction

A medical vision–language model can be accurate without being trustworthy. It can
answer correctly from a language prior while ignoring the image, and it can
answer confidently when the evidence needed is absent from the image entirely.
The conference version of this work formalised **evidence-conditional
trustworthiness** — a model is trustworthy when its answer is conditioned on
evidence actually present, and when it declines when that evidence is not — and
operationalised it as a seven-metric composite (MCS) over Capability, Safety and
Grounding, measured on 300 radiologist-supervised cases from four 2D medical VQA
sources.

That audit produced two results that this extension exists to follow. First, the
Visual Grounding Ratio separated models that consult the region of interest from
those that do not: **every model in the conference version's headline table is
positive**, from +3.6 to +49.5 pp. Second, a visual-information-decay sweep found that models
remain visually anchored through substantial degradation: the language-takeover
point $L^*$, the blur level at which residual visual contribution falls below
20 % of the clean-image contribution, lies between $\sigma = 16$ and $64$ pixels.
In 2D, the visual pathway is doing work, and it takes real degradation to stop it.

Both findings were measured on single images. Most diagnostic CT and MR work is
volumetric, and it was not obvious that either would survive the change — but
testing it required solving the problem that limits the 2D suite itself.

**The annotation bottleneck, and what replaces it.** In the 2D suite every probe
is authored and adjudicated by clinicians through a three-stage pipeline, so
probe count scales linearly in expert time. The **V-CF arm is the most expensive
part**: a visual counterfactual must state a condition under which the gold
answer flips, and a clinician must certify the flip. For volumes this is worse
than expensive — the counterfactual scan does not exist, so the flip cannot be
observed at all, only described.

This extension replaces that construction with a **deterministic anatomical
simulator**. Given an expert lesion mask and an automatic anatomy segmentation,
the probe *"if this lesion grew by 12 mm in every direction, would it contact the
aorta?"* has a gold answered exactly by one distance transform: contact occurs
iff the lesion–target surface gap is at most 12 mm. Two such probes whose growth
amounts straddle the true gap satisfy the V-CF build constraint
$\ell^*(\text{anchor})=\ell^*(\text{T-CF})\neq\ell^*(\text{V-CF})$ **by
construction rather than by adjudication**. The conference version's most
expensive probe family becomes free, verifiable to the voxel, and limited only by
compute.

**Contributions of this extension.**

1. **A simulator-backed V-CF construction** and a 9,484-probe / 588-volume /
   four-organ CT corpus built at zero annotation cost, with per-item provenance
   sufficient to recompute every gold independently.
2. **A volumetric audit of thirteen models across eight families** under the unchanged
   seven metrics, giving the first cross-modality reading of the same framework.
   **The Grounding axis that discriminated in 2D saturates or inverts in 3D**:
   VGR spans +3.6…+49.5 pp across capable 2D models against −8.3…+0.8 pp in 3D,
   is *exactly* +0.0 for one model in all four organs, and is negative in every
   organ for the two models whose response channel is verified. On confound-free
   items every model sits at the exactly-balanced chance level (48.0–50.6 %,
   all thirteen intervals containing 50) with image contributions from −3.1 to
   +0.7 pp — none above chance, none helped by the volume.
3. **Two controls the 2D suite did not require.** A known-answer *response-channel*
   control shows that **four of twelve audited models cannot answer "Is this a
   CT scan?" reliably** — all three released native volumetric medical VLMs and
   one general-purpose model, each near-perfect in one answer direction and
   failing in the other (61 %, 25 %, 3 % and 0 % correct on the questions their
   bias runs against). Their scores are uninterpretable, a failure mode invisible to
   accuracy since they sit at 50 % alongside models that fail for entirely
   different reasons — and not predictable from architecture, which is why it
   has to be measured rather than assumed. And *content-free option calibration*, without
   which likelihood scoring awards a perfect 0.0 % silent-failure rate to a model
   that fabricates on 97.3 % of hallucination traps, and collapses a 42-point
   composite difference between two models to 2.0.
4. **An intervention that works on one axis and not the other, and the checks
   that separated them.** A text-side cue in our own corpus produced a
   statistically impeccable *capability* gain that we withdraw, together with two
   findings that depended on it — every guard conventional in this literature
   passed it. Re-run on the safety axis, the same intervention cuts M3D-LaMed's
   weighted silent-failure rate by **17.4 pp** (95 % CI [−30.5, −9.1]) on the
   cleanest trap family, while leaving Qwen2.5-VL-7B's unmoved (−1.5 pp,
   [−5.0, +7.0]). Visual signal is present in the logits of the model that
   fabricates most and cannot be reached in the model whose response channel is
   clean — which is not the direction we expected, and is reported as measured.

**What does not fail.** The volumetric visual pathway is not inert, and two
independent measurements say so. Replacing the volume with zeros costs 4.3 to
8.4 accuracy points across four organs, so the scan is worth several points to a
model that reaches chance on the question it was asked. And amplifying the
sighted-versus-blind logit difference cuts M3D-LaMed's weighted silent-failure
rate by 17.4 pp, so the signal is not merely present but recoverable. What the
four-arm decomposition adds is *where* it is not: none of those points comes
from the lesion, the target, or the corridor between them (§7.5). The failure
this extension documents is a failure of volumetric *spatial reasoning*, not of
volumetric perception, and any remedy aimed at the encoder is aimed at the part
that already works.

> **Figure 1 (revise).** The conference version's motivation figure — silent
> failure vs. safe refusal — should gain a second panel showing the same
> contrast on a CT volume, with the counterfactual growth annulus drawn around
> the lesion and the target structure outlined. Panel (a) 2D as published,
> panel (b) 3D.

---

## 2. Related Work

### 2.1 Medical VQA benchmarks and the cost of their ground truth

Medical VQA was established in 2D by VQA-RAD [Lau et al., *Sci. Data* 2018] and
SLAKE [2102.09542]; the conference version draws on both, plus ROCO and CXR
collections. All are annotation-derived, so probe production costs expert time
and the space of probeable situations is bounded by the situations that occurred.

### 2.2 Volumetric medical VLMs

Native 3D medical VLMs form a small, architecturally diverse family: M3D-LaMed
[2404.00578] (3D ViT + Phi-3), Med3DVLM [2503.20047] (dcformer + Qwen2.5, a
different input geometry), RadFM [2308.02463], Merlin [2406.06512], CT-CHAT on
CT-RATE [2403.17834], and Med3D-R1 [2602.01200]. CT-Agent [2505.16229] is a
tool-augmented response to the same difficulty. We audit the two openly released
systems with runnable volumetric inference, chosen for architectural
disagreement, alongside four general-purpose models fed orthogonal views.

### 2.3 Volumetric spatial reasoning and its ground truth

CT-SpatialVQA [2605.08787] reports models "lost in volume"; DeepTumorVQA
[2605.09679] separates measurement from visual reasoning and locates the gap in
the latter. We agree with both and claim no priority on the observation. What
differs is the instrument: report-derived ground truth cannot express a
counterfactual, treats an unmentioned relation as unobserved rather than
negative, and couples question and answer through one human process.

### 2.4 Contrastive decoding, blind controls, segmentation

The amplification intervention of §5.4 descends from contrastive decoding
[2210.15097], visual contrastive decoding [2311.16922] and classifier-free
guidance [2207.12598]. Anatomy is segmented with TotalSegmentator [2208.05868].

---

### 2.5 Auditing what a model conditions on, and why 3D changes the question

Work on whether a vision–language model uses its image divides into three
approaches, and the volumetric setting stresses each differently.

**Ablate the input and compare.** Blind baselines, region masking and the
Visual Grounding Ratio all measure a difference between two input conditions
[MedVIGIL; 2401.06209; 2310.14566]. The approach assumes the two conditions
differ only in the evidence. In 2D that is close to true — an ROI crop of a
chest radiograph is most of the radiograph. In 3D it is false: a region around
a lesion and its target is a fraction of a percent of the volume, so the two
arms differ in how much image survives as much as in what evidence remains.
§7.5 measures both and finds the second term dominant, which is why we report a
four-arm decomposition rather than a ratio.

**Probe with counterfactuals.** Asking about states that did not occur defeats
retrieval from priors [2210.15097; 2311.16922], and is the strategy this work
extends. Its cost is that the counterfactual must be answerable, which in 2D
means a clinician adjudicates each one. Computing the answer from geometry
removes that cost and, as §7.6 documents, introduces a different one: a
generated corpus can carry a cue that no annotated corpus would.

**Measure the response channel before scoring it.** Known-answer controls are
routine in psychophysics and rare in VLM evaluation, where models are assumed to
answer questions. Four of our thirteen models do not, and the four are not
predictable from architecture (§7.3). Any 3D audit that omits this control
reports the scores of models that are not answering.

The 2D medical VQA literature also assumes that likelihood over answer strings
is a fair scoring rule. With two one-token options it is. With heterogeneous
options — one of which is a refusal — it is not, and the failure is total rather
than gradual: every model in this audit scores exactly 0.0 % weighted
silent-failure before calibration and 43.7–96.4 % after (§7.7).

## 3. Evidence-Conditional Trustworthiness

Unchanged from the conference version except where the volumetric instantiation
is stated.

### 3.1 Formal setup

Let $f$ be a model, $x$ an image, $q$ a question, $\ell^*$ the gold. Evidence-
conditional trustworthiness requires that $f(x,q)$ track $\ell^*(x,q)$ *because
of* evidence in $x$, and that $f$ select refusal when the premise of $q$ is
unsatisfiable in $x$.

In the volumetric instantiation $x$ is a volume $V$ in RAS+ orientation with
spacing $s\in\mathbb{R}^3_{>0}$, an expert lesion mask $L$, and automatic anatomy
$\{A_k\}$. All geometry is computed in millimetres; CT slice spacing routinely
differs from in-plane spacing by a factor of four, so a voxel-space radius is a
different physical quantity in every scan.

### 3.2 Probe families and output taxonomies

Five families, unchanged in definition:

| Family | Construction | Gold |
|---|---|---|
| Anchor | original (image, question) | as annotated (2D) / as computed (3D) |
| T-CF | question paraphrased, semantics preserved | preserved |
| NEG | question logic inverted | flipped |
| SDR | qualifier removed, answer unaffected | preserved |
| Trap | premise unsatisfiable in this image | refusal |

Every probe is a five-option MCQ with an explicit refusal option; the output
taxonomy (correct / incorrect / refusal) is unchanged.

### 3.3 One framework, two realisations of V-CF

This is the only substantive methodological change, and it is a change of
*realisation*, not of definition.

| | 2D (conference) | 3D (this extension) |
|---|---|---|
| V-CF condition | textual condition added to the same image | growth amount $G$ stated in mm |
| Gold flip | adjudicated by clinician | $\ell^* = [\,d(L,A_k)\le G\,]$, computed |
| Build constraint | enforced by review | satisfied by construction |
| Marginal cost | expert time | one distance transform per lesion |

$$d(L,A_k)=\min_{u\in L,\;v\in A_k}\lVert (u-v)\odot s\rVert_2$$

A single Euclidean distance transform of $\neg L$ under spacing $s$ answers every
growth query for that lesion exactly, so generation costs $O(1)$ transforms per
lesion rather than $O(|A|)$.

**Refuse when undecidable.** Structures overlapping on the queried axis, and
lesions straddling two containers, yield no probe rather than a forced label
(11.1 % of lung lesions refused). Distractors are capped at a 6× gap ratio. Every
directional item ships its computed margin, and a provenance gate re-derives the
gold from that geometry alone, failing the build on mismatch (0 errors over 8,476
items).

---

## 4. Datasets

### 4.1 Evaluative claims and assumptions

Two assumptions are specific to the volumetric instantiation, and both are
load-bearing.

**A1. Geometric gold is clinically meaningful** — that a radiologist would answer
the contact question the same way. This is what §7.11 exists to test and the one
assumption computation cannot discharge.

**A2. Counterfactual construction defeats language priors.** The 2D suite can
rely on this because its V-CF condition is authored to be non-inferable. **We
assumed it for the geometric construction and were wrong** (§7.6). We now treat
it as a claim that must be measured on the exact items being graded.

### 4.2 Source mix and risk coverage

| | 2D (conference) | 3D (this extension) |
|---|---|---|
| Sources | VQA-RAD 120, SLAKE 60, ROCO 60, CXR 60 | MSD liver / lung / pancreas / colon |
| Units | 300 cases | 588 volumes |
| Probes | 2,556 MCQ | 9,484 (3,200 in family form) |
| Counterfactuals | 240 triplets | 4,238 matched pairs |
| Annotation cost | three-stage clinician pipeline | zero |
| CRT L1:L2:L3:L4:L5 | 71:31:118:43:37 | assigned by target structure |

Clinical-risk tiers and harm weights are inherited unchanged
($w=1,2,3,5,8$). In 3D the tier follows the *target structure*: aorta, heart and
spinal cord are L5; major organs L4; presence-of-finding L3 by default;
vertebrae L2; sternum and ribs L1.

### 4.3 Three-stage pipeline: annotation versus generation

| Stage | 2D: annotation | 3D: generation |
|---|---|---|
| 1 | clinician authors probe from a case | simulator enumerates $(L,A_k)$, computes $d(L,A_k)$ |
| 2 | second clinician adjudicates gold and flip | provenance gate re-derives gold independently |
| 3 | senior review; risk-tier assignment | tier by target; refuse-when-undecidable rules |

Stages 2 and 3 become assertions executed at build time. **What is not replaced
is validation**: A1 is tested by an independent reader study (§7.11), and the
annotation-free claim is defensible only if that test passes.

> **Figure 2 (revise).** The construction-and-evaluation pipeline figure should
> become two parallel tracks sharing one evaluation block: the published 2D
> annotation track on the left, the 3D generation track on the right, converging
> on the same paired-evaluation and metric computation. This figure carries the
> paper's premise and is worth the space.

---

## 5. Metrics

### 5.1 Closed-loop evaluation pipeline

Inherited. Scoring is by likelihood over option strings, not single-letter
parsing at $T=0$: volumetric medical VLMs do not reliably emit the letter format
— M3D-LaMed answers in referring-expression templates ("The object in question is
Paris") whose wording embeds answer words ("The object is rib **left** 5"), so
text parsing credits words the model never chose.

**Addition 1: content-free option calibration.** Likelihood scoring is unbiased
for two one-token options ("yes"/"no"). Across five options of unequal length it
is dominated by the prior plausibility of the option *string*. We score

$$\text{score}(c)=\log P(c\mid q,x)-\log P(c\mid \texttt{"N/A"},x)$$

keeping the image in both terms, so the option-string prior is removed without
removing the image contribution. The correction depends only on
(volume, option set, option) — ten forward passes per volume calibrate every probe
on it, applied to stored likelihoods with no re-evaluation. §7.7 shows this is
not a refinement but a precondition: uncalibrated the Safe axis is awarded to a
constant and both composites collapse to the floor, hiding a 42-point difference.

**Addition 2: response-channel controls.** Before interpreting any accuracy, each
model answers seven questions with externally known answers about the same
volumes, through the identical scoring path. The 2D suite did not require this.
§6.2 shows the 3D suite does.

### 5.2 Per-family metric definitions

Inherited verbatim: Acc_orig, PR, NEG, SDR are family accuracies; SFR is the
fraction of trap responses selecting a non-refusal option; LPA is accuracy on
image-independent knowledge-only probes; and

$$\mathrm{VGR}=\mathrm{Acc}_{\text{ROI-only}}-\mathrm{Acc}_{\text{ROI-masked}},
\qquad
\mathrm{SFR}_w=\frac{\sum_k w_k\,\mathrm{SFR}_{L_k}}{\sum_k w_k}.$$

**ROI regions are computed, not drawn.** The conference version notes its ROIs
are benchmark regions for binary masks rather than segmentation labels; in 3D the
evidence region — lesion, target, and the corridor between them, dilated 15 mm —
comes from the same geometry that produced the gold, so inter-rater IoU is not a
meaningful statistic for it. Masked background is filled with the volume's own
1st percentile rather than zero, so the manipulation does not announce itself as
an out-of-distribution value.

### 5.3 The composite score

Inherited verbatim:

$$\mathrm{Cap}=\tfrac14[\mathrm{Acc_{orig}}+\mathrm{PR}+\mathrm{NEG}+\mathrm{SDR}],\qquad
\mathrm{Safe}=100-\mathrm{SFR}_w,$$
$$\mathrm{Ground}=\tfrac12[\mathrm{clip}(\mathrm{VGR}+50,0,100)+\mathrm{Acc_{ROI\text{-}masked}}],$$
$$\mathrm{MCS}=\frac{3\,\mathrm{Cap}\cdot\mathrm{Safe}\cdot\mathrm{Ground}}
{\mathrm{Cap}\cdot\mathrm{Safe}+\mathrm{Cap}\cdot\mathrm{Ground}+\mathrm{Safe}\cdot\mathrm{Ground}}$$

### 5.4 Contrastive visual amplification

$$\mathrm{logit}_\alpha(c)=\mathrm{logit}_{\text{blind}}(c)+\alpha\big[\mathrm{logit}_{\text{sighted}}(c)-\mathrm{logit}_{\text{blind}}(c)\big]$$

with $\alpha$ selected out-of-fold over patients, intended as a diagnostic
separating "visual signal present but out-voted" from "no usable visual signal".
§7.8 reports where its result holds and where it does not.

**Statistics.** All intervals are patient-level bootstrap over volumes, never
item-level: 9,484 probes come from 588 volumes. Intervals on the intervention
resample the *entire* cross-validated procedure, re-selecting $\alpha$ inside each
resample; resampling a fixed $\alpha$ understates the spread and can place the
point estimate outside its own interval.

---

## 6. Audit I — 2D: annotation-backed evidence

*Conference-version results, reproduced here so that the cross-modality analysis
of §8 stands on one document. Numbers are unchanged.*

### 6.1 Setup

18 vision-capable models plus an independent, construction-blind radiologist
baseline (R4), on 300 cases and 2,556 MCQ probes.

### 6.2 Headline results

| Model | Acc_orig | PR | NEG | SDR | SFR↓ | SFR$_w$↓ | VGR | LPA | Cap | Safe | Ground | MCS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Radiologist R4 (ref.)** | 95.3 | 92.7 | 90.7 | 91.5 | **5.8** | 9.3 | +4.5 | 93.6 | 92.6 | 90.7 | 70.5 | **83.3** |
| Claude Opus-4.7 | 78.7 | 78.0 | 83.1 | 79.9 | 22.0 | 25.2 | +22.9 | 98.6 | 79.9 | 74.8 | 57.3 | **69.2** |
| Gemini 3.1-Flash-Lite | 73.0 | 75.7 | 63.1 | 70.7 | 22.3 | 26.6 | +49.5 | 98.9 | 70.6 | 73.4 | 56.5 | 66.0 |
| Gemini 3-Flash | 77.0 | 80.7 | 75.6 | 76.2 | 37.2 | 42.6 | +41.1 | 98.9 | 77.4 | 57.4 | 59.6 | 63.7 |
| GPT-5.5 | 75.0 | 74.3 | 68.9 | 59.8 | 36.8 | 39.5 | +12.0 | 99.6 | 69.5 | 60.5 | 55.5 | 61.3 |
| GPT-5.4 | 68.0 | 68.7 | 56.4 | 59.8 | 28.8 | 32.7 | +15.6 | 98.6 | 63.2 | 67.3 | 47.4 | 57.9 |
| Claude Sonnet-4.6 | 63.3 | 66.3 | 60.9 | 56.7 | 41.3 | 46.9 | +18.2 | 97.9 | 61.8 | 53.1 | 46.1 | 52.9 |
| HuatuoGPT-Vision-7B | 55.3 | 59.7 | 26.2 | 51.2 | 31.5 | 39.6 | +28.1 | 91.9 | 48.1 | 60.4 | 45.1 | 50.4 |
| LLaVA-Med | 43.3 | 50.0 | 15.6 | 50.6 | 43.0 | 46.7 | +28.6 | 75.6 | 39.9 | 53.3 | 43.0 | 44.7 |
| GPT-4o | 59.3 | 63.3 | 45.8 | 50.6 | 53.0 | 64.0 | +3.6 | 98.6 | 54.8 | 36.0 | 45.3 | 44.1 |
| GPT-5.4-mini | 55.3 | 57.0 | 44.9 | 47.0 | 49.7 | 62.2 | -8.9 | 98.2 | 51.0 | 37.8 | 41.9 | 42.9 |
| Qwen3.5-397B-A17B | 53.0 | 53.0 | 33.3 | 42.7 | 54.3 | 64.6 | +24.0 | 92.2 | 45.5 | 35.4 | 40.1 | 39.9 |
| Claude Haiku-4.5 | 46.7 | 51.0 | 32.4 | 42.1 | 49.0 | 58.4 | -9.4 | 97.5 | 43.0 | 41.6 | 35.7 | 39.8 |
| Kimi-K2.6 | 49.3 | 47.7 | 36.0 | 37.8 | 50.8 | 61.5 | -2.1 | 95.4 | 42.7 | 38.5 | 35.7 | 38.8 |
| Kimi-K2.5 | 49.7 | 54.3 | 36.4 | 41.5 | 60.0 | 68.9 | +10.4 | 95.1 | 45.5 | 31.1 | 41.1 | 38.2 |
| GPT-5.4-nano | 31.7 | 29.7 | 16.9 | 25.0 | 51.7 | 58.2 | -27.1 | 90.5 | 25.8 | 41.8 | 31.0 | 31.6 |
| Qwen3.5-9B | 43.3 | 48.0 | 28.4 | 32.3 | 62.8 | 76.1 | +12.5 | 84.5 | 38.0 | 23.9 | 34.9 | 31.0 |
| DeepSeek-V4-Flash | 4.0 | 4.3 | 6.7 | 4.9 | 42.7 | 50.9 | -86.5 | 99.3 | 5.0 | 49.1 | 43.8 | 12.3 |
| DeepSeek-V4-Pro | 1.7 | 5.0 | 6.2 | 1.8 | 37.8 | 47.6 | -96.4 | 99.3 | 3.7 | 52.4 | 48.7 | 9.6 |

*All 18 audited models, recomputed from the released per-model metrics.* Table 5
of the conference version reports the **unweighted** trap rate in its SFR column;
the Safe axis of the composite uses the **harm-weighted** SFR$_w$, and we list
both. Recomputing Claude Opus-4.7 with the unweighted value gives MCS 70.1
against a published 69.2, and with SFR$_w$ = 25.2 it reproduces 69.2 exactly.
The recomputation asserts itself against the published values rather than
claiming agreement: `published2d.csv` holds them and the build fails if any row
drifts by more than 0.05. It currently pins the strongest audited model (69.2)
and the reader reference (83.3), which between them exercise the composite
formula, the harm weighting and both ends of the range. The volumetric audit
uses SFR$_w$ throughout, so the two modalities are comparable without
adjustment.

The R4 row reproduces from the same formula and the same weighting as every
model row: Cap 92.6 from its four family accuracies, Ground 70.5 from its ROI
arms (91.0 only, 86.5 masked, VGR +4.5), and Safe 90.7 — that is, a *weighted*
SFR of 9.3 against its 5.8 unweighted, exactly the direction the weighting takes
every model. These give **MCS 83.3**, and 83.3 − 69.2 = 14.1, the composite
headroom above the strongest audited model.

Three properties of this table matter for what follows.

**The axis discriminates.** Every model in the conference version's headline
table has positive VGR, from +3.6 (GPT-4o) to +49.5 (Gemini 3.1-Flash-Lite): in
2D the evidence region demonstrably carries the answer.

Extending the recomputation to all 18 released model results adds six negative
entries, none of which appears in that table. Two are DeepSeek-V4 variants that
answer almost nothing correctly under either ROI arm (Acc_orig 1.7–4.0 %), which
the composite handles by collapsing Cap rather than by crediting grounding; the
other four (GPT-5.4-mini −8.9, Claude Haiku-4.5 −9.4, Kimi-K2.6 −2.1,
GPT-5.4-nano −27.1) are smaller variants that do no better with the evidence
region than without it. What carries into §8 is the *spread*: an axis running
+3.6…+49.5 among capable 2D models has a 6.1-point range in 3D and never exceeds
+0.8.

**Language priors are near-ceiling.** LPA is 91.9–99.6 % for every model except
LLaVA-Med, and *above the radiologist's own* 93.6 % for most. Models hold the
knowledge; the audit's discriminating power comes from Safety and Grounding, not
from knowledge.

**Safety is the gap to human performance.** R4's SFR is 5.8 %; the best model is
22.0 %, and the median is above 40 %. The reader leads on all three axes — Cap 92.6 against 79.9, Safe 90.7 against
74.8, Ground 70.5 against 57.3 — and the **14.1-point** composite headroom above
the strongest audited model is predominantly a Safety headroom.

### 6.3 Ablation: visual information decay

Gaussian blur at $\sigma\in\{0,2,4,8,16,32,64\}$ pixels plus a no-image control;
the language-takeover point $L^*$ is the smallest $\sigma$ at which residual
visual contribution falls below 20 % of the clean-image contribution.

| Model | $\sigma$=0 | 2 | 4 | 8 | 16 | 32 | 64 | no-image | clean contrib. | $L^*$ |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 59.0 | 58.7 | 50.3 | 46.3 | 34.7 | 19.0 | 15.3 | 5.3 | 53.7 | **64** |
| Claude Sonnet-4.6 | 65.7 | 65.0 | 56.3 | 47.7 | 31.0 | 14.7 | 4.7 | 9.3 | 56.3 | 32 |
| HuatuoGPT-V-7B | 55.3 | 56.0 | 52.7 | 43.0 | 28.3 | 17.7 | 13.3 | **20.7** | 34.7 | 32 |
| Qwen3.5-397B | 50.3 | 49.7 | 48.0 | 36.0 | 17.7 | 8.3 | 7.7 | 10.0 | 40.3 | **16** |

Models remain visually anchored through substantial degradation: accuracy is
essentially unchanged at $\sigma=2$ and still well above the no-image floor at
$\sigma=8$. HuatuoGPT-V has the highest no-image floor (20.7 %, four times
GPT-4o's), i.e. the strongest language prior among the medical-tuned models.

---

## 7. Audit II — 3D: simulator-backed evidence

### 7.1 What this audit tests, and where each part is tested

Evidence-conditional trustworthiness has two halves, and the volumetric audit
tests them separately because they can fail separately.

**Is the answer conditioned on evidence that is present?** §7.4 asks whether
accuracy exceeds chance once the text-side cue is removed; §7.5 asks whether the
region the answer geometrically depends on is the region the model uses; §7.10
asks whether the volume reaches the decision at all, and separates "arrives and
is out-voted" from "arrives and rewrites the answer wrongly". §7.6 documents the
cue that made this harder to see, in our own corpus.

**Does the model decline when the evidence is not present?** §7.7 puts
unsatisfiable premises to it — structures verified absent from that specific
volume — and asks whether seeing the scan changes the refusal.

Two prerequisites gate both halves. §7.3 checks that the model's output channel
can express an answer at all, without which no score is interpretable; §7.9
checks how far the language prior alone gets, which bounds what any visual
result can mean.

The short answer is that both halves fail, and that they fail for different
reasons in different models — which is the case for reporting the axes
separately rather than as one composite.

### 7.2 Setup

Thirteen models spanning eight families and both input routes. Fed
orthogonal-view montages: SmolVLM2-2.2B, Qwen2.5-VL 3B / 7B / 32B,
Qwen3-VL-8B, InternVL3 8B / 14B, LLaVA-OneVision-7B, Idefics3-8B and
Pixtral-12B. Taking volumes natively: M3D-LaMed-Phi3-4B, M3D-LaMed-Llama2-7B
and Med3DVLM-7B. The two M3D variants share a vision tower and differ only in
backbone, which separates the two candidate locations for a volumetric failure;
the native systems were otherwise chosen for architectural disagreement —
different vision towers, different backbones, and opposite volume-axis
conventions, $(D,H,W)$ against $(H,W,T)$.

**Two models could not be run, and one of them is worth reporting.**
HuatuoGPT-Vision-7B, audited in 2D, has a `llava_qwen2` architecture unrecognised
by the installed transformers release, and the repository ships no modeling code:
it fails at load, visibly. Aria-25B-MoE fails differently. Its repository
declares a processor file it does not contain, so the remote-code path dies; the
natively implemented path loads, but the checkpoint is keyed
`language_model.model.layers.*` against an implementation expecting
`model.language_model.*`, so **338 language-model weight tensors are newly
initialised** and the load succeeds with a warning. That model runs, emits
logits, and produces a benchmark score — of a randomly initialised language
model, which on this corpus would look like one more system at chance and would
have been reported as such. We added a load-time check that refuses any model
whose language-model weights the checkpoint did not fill, and report the model
as excluded rather than as a result.

Growth-contact labels are balanced by construction, so chance is exactly 50.0 % —
but see §7.6, where we show this bounds the *label* distribution and not what a
text-only model can score.

**Is the montage the problem?** The obvious objection to feeding a volume to a
2D model as an orthogonal-view montage is that the interface, not the model, is
what fails. Three measurements answer it, and none of them depends on the models
being right about anything.

- **The montage delivers information that reaches the decision.** Shown the
  volume rather than a blank one, InternVL3-14B changes its answer on 92.0 % of
  probes, Qwen3-VL-8B on 77.4 %, Qwen2.5-VL-32B on 48.9 % (§7.10). An interface
  that carried nothing could not rewrite most of a model's answers.
- **The montage is worth several accuracy points.** Replacing the volume with
  zeros costs 4.3 to 10.4 points across four organs and two models (§7.5). The
  models extract usable signal from it; they merely extract nothing that answers
  the question asked.
- **Native volumetric input does not rescue the result.** Three models take the
  volume directly, with no montage: M3D-LaMed-Phi3-4B, M3D-LaMed-Llama2-7B and
  Med3DVLM-7B sit at 50.1 %, 48.0 % and 50.0 % on confound-free items. Nor does
  the montage explain the response-channel failures, since one montage-fed model
  fails them and seven do not (§7.3).

The three native systems cannot carry the claim on their own, because §7.3 shows
their response channels are unreliable. But the objection is not that native
models would do *worse*; it is that the montage suppresses a capability the
volume would reveal. Nothing in these three suggests it does, and the first two
measurements show the montage is not suppressing the signal — it is delivering
signal the model then fails to use for geometry.

### 7.3 Response-channel controls: two systems that cannot be scored

Seven known-answer questions × 20 volumes × 6 models, through the identical
scoring path. The questions carry no clinical content: a model that cannot answer
*"Is this a CT scan?"* about a CT scan has not failed at radiology.

| Model | Input | Passed | "yes" rate | yes-gold | no-gold | Margin |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | montage | **140/140** | 42.9 % | 100 % | 100 % | 2.90 |
| Qwen2.5-VL-7B | montage | **140/140** | 42.9 % | 100 % | 100 % | 7.36 |
| Qwen2.5-VL-32B | montage | **140/140** | 42.9 % | 100 % | 100 % | 8.62 |
| InternVL3-8B | montage | **140/140** | 42.9 % | 100 % | 100 % | 10.19 |
| InternVL3-14B | montage | **140/140** | 42.9 % | 100 % | 100 % | 8.31 |
| Qwen3-VL-8B | montage | **140/140** | 42.9 % | 100 % | 100 % | 19.26 |
| M3D-LaMed-Phi3-4B | native 3D | 107/140 | 63.6 % | 97 % | 61 % | 8.28 |
| M3D-LaMed-Llama2-7B | native 3D | 80/140 | 85.7 % | 100 % | 25 % | 1.81 |
| Med3DVLM-7B | native 3D | 80/140 | 0.0 % | 0 % | 100 % | 15.32 |

**Four of thirteen models answer from a fixed direction rather than from the
image, and the four are not the four an architectural reading would predict.**
Seven models answer all seven control questions perfectly — 7/7 at 20/20, "yes"
rate 42.9 %, exactly the 3/7 the set implies. SmolVLM2-2.2B answers six of seven
and fails one lexically at a still-balanced 56.4 %. The remaining four fail by
polarity: they are near-perfect on the questions whose answer is "yes" and fail
the ones whose answer is "no", or the exact reverse.

| Model | Input | Pass | gold = yes | gold = no | Direction |
|---|---|---|---|---|---|
| seven models (Qwen2.5-VL 3B/7B/32B, Qwen3-VL-8B, InternVL3 8B/14B, LLaVA-OneVision, Pixtral) | montage | 140/140 | 100 % | 100 % | — |
| SmolVLM2-2.2B | montage | 121/140 | 100 % | 82 % | one phrasing |
| M3D-LaMed-Phi3-4B | native | 107/140 | 97 % | **61 %** | yes-biased |
| M3D-LaMed-Llama2-7B | native | 80/140 | 100 % | **25 %** | yes-biased |
| Idefics3-8B | montage | 82/140 | **3 %** | 98 % | no-biased |
| Med3DVLM-7B | native | 80/140 | **0 %** | **100 %** | no-biased, absolute |

**One of these numbers was ours, not the model's, and we report the correction
because it changes what the section claims.** The control originally framed every
question with "This is a CT scan shown as axial, coronal and sagittal views" —
true of a montage and false of a volume, and it answers one of the control
questions outright. Re-run with a framing matching each model's actual input,
M3D-LaMed-Phi3 goes from 66/140 to 107/140 and its "yes" rate from 95.7 % to
63.6 %; its accuracy on negative-answer questions rises from 8 % to 61 %. An
earlier version of this section said that none of the three native systems could
answer "Is this a CT scan?" reliably. Under a framing that does not describe an
input they never received, two of them answer it 18/20 and 20/20. That claim was
an artefact of our harness and we withdraw it.

**What survives the correction is sharper than what it replaces.** The failures
are not an inability to see — every failing model is at or near 100 % on the
questions asking whether something *is* present. They are a refusal to answer in
one direction. M3D-LaMed-Llama2 is correct on 100 % of yes-answer questions and
25 % of no-answer ones; Med3DVLM-7B is correct on 100 % of no-answer questions
and **0 %** of yes-answer ones, identically under both framings, so its bias is
not a prompt effect at all.

**And it is the same failure the safety axis measures.** A model that cannot
answer "no" to *is this a photograph of a cat* is the model that cannot answer
"the gallbladder is not in this scan" when asked whether a lesion would reach it
— it fabricates rather than declines (§7.7). The response-channel control and the
hallucination traps were designed as separate instruments; under a correct
framing they turn out to measure one thing from two directions.

What the split does not follow is scale (2.2B through 32B on both sides of it),
model family (eight families, with failures in two), or decision margin: Qwen3-VL-8B
passes with a margin of 19.26, more than twice M3D-LaMed-Phi3's 8.28, and
M3D-LaMed-Llama2 fails at 1.81. A wide margin is not pathological; a wide margin
on the wrong answer is.

This is the paper's least contestable result. It uses a separate question set
with no growth parameter, so §7.6's confound cannot reach it; both options are
one token, so §5.1's calibration is not needed; and the proportions are far
enough apart that no test is required to read them.

**The two M3D variants isolate where the failure lives.** They share the
vision tower and the input pipeline and differ in the LLM backbone (Phi-3 4B
against Llama-2 7B). Both fail the same way — biased toward "yes", correct on
7.5 % and 20.0 % of no-gold controls — at *decision margins that differ by a
factor of 4.6* (8.28 against 1.81). The failure therefore does not follow the
backbone, and it is not a consequence of over-confidence: one variant is
over-confident and the other is not, and both answer the same controls wrongly.
That points at the vision–language interface these two share, which is the
comparison the second variant was added to make.

The degeneracies differ in kind. Med3DVLM's is an unconditional constant:
"no" to all 140 controls and all 2,262 probes. M3D-LaMed's is **phrasing-driven
and sign-flipping** — mean $\log P(\text{yes})-\log P(\text{no})$ is **+7.83** on
controls and **−6.57** on benchmark questions, where it answers "no" 1,902 times
against "yes" 360. Its answer is set by the surface form of the question, with a
margin large enough that image evidence cannot overturn it.

**Consequence.** Only the four passing models support claims about spatial
reasoning, and those claims are thereby strengthened: a model answering 140/140
control questions correctly and then performing at exactly chance with ≈0 image
contribution has failed at *the task*, not at responding. Results computed on
M3D-LaMed characterise a system whose binary channel is prior-dominated, and are
labelled as such.

Read from task accuracy alone, all six sit at ~50 % and look like one phenomenon.
They are at least three. **This control was not needed in 2D and is
indispensable in 3D**; we recommend it precede any capability number in the
volumetric regime.

### 7.4 Headline results

| Model | Input | Full corpus | **Growth-matched** | 95 % CI | Matched, blind | Image gain | Pair viol. |
|---|---|---|---|---|---|---|---|
| SmolVLM2-2.2B | montage | 49.9 % | **50.0 %** | [48.2, 51.8] | 50.0 % | +0.0 | 99.8 % |
| Qwen2.5-VL-3B | montage | 50.0 % | **50.0 %** | [48.2, 51.8] | 50.0 % | +0.0 | 99.4 % |
| Qwen2.5-VL-7B | montage | 50.0 % | **50.0 %** | [48.2, 51.8] | 50.0 % | +0.0 | 100.0 % |
| Qwen2.5-VL-32B | montage | 56.0 % | **50.6 %** | [48.6, 52.6] | 49.9 % | +0.7 | 86.7 % |
| InternVL3-8B | montage | 58.4 % | **50.1 %** | [48.0, 52.2] | 50.0 % | +0.1 | 79.4 % |
| InternVL3-14B | montage | 53.0 % | **50.1 %** | [48.4, 51.9] | 49.5 % | +0.7 | 94.1 % |
| Qwen3-VL-8B | montage | 55.6 % | **48.7 %** | [46.6, 50.7] | 50.0 % | −1.3 | 88.8 % |
| LLaVA-OneVision-7B | montage | 50.1 % | **49.5 %** | [47.7, 51.3] | 50.0 % | −0.5 | 99.8 % |
| Idefics3-8B | montage | 50.0 % | **50.0 %** | [48.2, 51.8] | 50.0 % | +0.0 | 100.0 % |
| Pixtral-12B | montage | 53.3 % | **51.0 %** | [48.8, 53.3] | 50.0 % | +1.0 | 86.6 % |
| M3D-LaMed-Phi3-4B | native | 50.5 % | **49.9 %** | [48.1, 51.7] | 50.0 % | −0.1 | 93.9 % |
| M3D-LaMed-Llama2-7B | native | 49.4 % | **50.1 %** | [48.2, 52.0] | 49.9 % | +0.1 | 93.7 % |
| Med3DVLM-7B | native | 50.0 % | **50.0 %** | [48.2, 51.8] | 50.0 % | +0.0 | 100.0 % |

Thirteen models, eight families, 2.2B to 32B, both input routes —
**48.7 % to 51.0 %**, all thirteen intervals containing 50, image contributions
−1.3 to +1.0 pp. No model is above chance and none is helped by the volume.

The three native rows are re-measured. Their first runs framed every probe as
"shown as axial, coronal and sagittal views", which describes an input they do
not receive; on the response-channel controls that framing cost M3D-LaMed-Phi3
41 of 140 questions (§7.3). Here it costs almost nothing — but under it
M3D-LaMed-Llama2 read as the only model with a significantly *negative* image
contribution (−3.1 pp), an anomaly that disappears when the framing is correct.
The framing interferes with questions about the image's own properties and not
with counterfactual geometry.

The three systems that look competent on the full corpus (55.6 %, 56.0 %, 58.4 %)
are exactly the three with the lowest pair-violation rates (88.8 %, 86.7 %,
79.4 %), and all three fall to chance once the growth cue is removed. Across thirteen
models the correspondence is monotone: distinguishing pair members means
comparing the two numbers, and doing so inflates full-corpus accuracy without
touching confound-free accuracy.

**Two distinct failures are pooled in that column, and they should be read
separately.** Prediction distributions over the 2,262 probes:

| Model | predictions | modal share |
|---|---|---|
| Qwen2.5-VL-7B | 2,262 "no" / 0 "yes" | **100.0 %** |
| SmolVLM2-2.2B | 2,260 "no" / 2 | 99.9 % |
| **LLaVA-OneVision-7B** | **2,236 "yes" / 26** | **98.9 %** |
| Qwen2.5-VL-3B | 2,251 / 11 | 99.5 % |
| M3D-LaMed-Phi3-4B | 1,902 / 360 | 84.1 % |
| InternVL3-8B | 1,577 / 685 | 69.7 % |
| **Qwen2.5-VL-32B** | **1,150 / 1,112** | **50.8 %** |

Qwen2.5-VL-7B emits a **constant** on this question form — every one of 2,262
probes answered "no" — while answering the response-channel controls 140/140 with
a 42.9 % "yes" rate. It is not a degenerate model; it is a model that declines to
engage with *this* question form.

The constancy is not an attraction to a particular token. **LLaVA-OneVision-7B
fails the same way in the opposite direction**, answering "yes" to 2,236 of
2,262 probes, and reaches the same place: 49.5 % on confound-free items, 99.8 %
pair violation, −0.5 pp from the image. Both models have a demonstrably working
response channel: each answers 7/7 control questions perfectly at a 42.9 % "yes"
rate, the balance the control set implies. Two models with verified, balanced
channels, two opposite constants, one outcome. What this question form fails to
elicit is a *decision*; which way a model defaults when it does not make one is
incidental, and neither default can be blamed on a broken output channel. That single fact explains three results we had
stated separately: its VGR is exactly +0.0 (§7.5), its decay curve is identical at
all eight conditions (§8.2), and its pair-violation rate is exactly 100.0 %. All
three are arithmetic consequences of constancy, and we now say so once rather
than three times.

**A second pairing confirms the constancy is not about numbers.** The
growth-contrast pair varies one integer, so a constant answer could in principle
mean "cannot compare two numbers". We therefore built a **target-contrast** pair:
same lesion, same growth amount, two different structures, one inside that
distance and one outside (600 pairs; growth medians identical at 20.9 mm for both
labels, so a threshold on the number gives exactly 50.0 %). Qwen2.5-VL-7B answers
**"no" to all 1,200 probes, sighted and blind alike**. Changing what varies from a
number to an anatomical name does not change the response, so the constancy is a
property of the question form rather than of numeric comparison.

**Qwen2.5-VL-32B is therefore the load-bearing case.** Its modal share is 50.8 %,
i.e. it genuinely differentiates probe from probe; it passes 140/140 controls;
and it still lands at **50.6 %** [48.6, 52.6] on confound-free items with an image
contribution of +0.7 pp. A model that engages with the questions, distinguishes
between them, and reaches chance is a stronger demonstration than one that
answers everything the same way. Its full-corpus 56.0 % came from the growth cue
(§7.6), not from the volume. The target-contrast pairs make the same point in
its sharpest form: shown the volume, Qwen-32B differentiates half of all pairs,
having differentiated none of them blind — and picks the correct direction
56.5 % of the time, indistinguishable from chance once the three models tested
are corrected together (§7.8).

Per-organ, on the **full** per-organ corpus (8,906 items per model, distinct
from the 2,262-item common subset above), split by probe kind. The two kinds
carry different chance levels and must not be pooled:

| Organ | Model | Probe kind | Chance | n | Sighted | Blind | Δ | Pair viol. |
|---|---|---|---|---|---|---|---|---|
| Lung | M3D | binary | 50 % | 462 | 50.6 | 51.7 | -1.1 | 94.4 % |
| Lung | M3D | 4-option | 25 % | 45 | 28.9 | 24.4 | +4.4 | — |
| Lung | Qwen | binary | 50 % | 462 | 50.0 | 50.0 | +0.0 | 100.0 % |
| Lung | Qwen | 4-option | 25 % | 45 | 17.8 | 26.7 | -8.9 | — |
| Colon | M3D | binary | 50 % | 752 | 53.2 | 49.9 | +3.3 | 92.0 % |
| Colon | M3D | 4-option | 25 % | 67 | 10.4 | 11.9 | -1.5 | — |
| Colon | Qwen | binary | 50 % | 752 | 50.0 | 50.0 | +0.0 | 100.0 % |
| Colon | Qwen | 4-option | 25 % | 67 | 20.9 | 23.9 | -3.0 | — |
| Pancreas | M3D | binary | 50 % | 4,608 | 50.4 | 49.7 | +0.7 | 92.7 % |
| Pancreas | M3D | 4-option | 25 % | 153 | 24.8 | 25.5 | -0.7 | — |
| Pancreas | Qwen | binary | 50 % | 4,608 | 50.0 | 50.0 | +0.0 | 100.0 % |
| Pancreas | Qwen | 4-option | 25 % | 153 | 32.7 | 29.4 | +3.3 | — |
| Liver | M3D | binary | 50 % | 2,654 | 51.3 | 52.2 | -0.9 | 92.1 % |
| Liver | M3D | 4-option | 25 % | 165 | 26.1 | 23.0 | +3.0 | — |
| Liver | Qwen | binary | 50 % | 2,654 | 50.0 | 50.0 | +0.0 | 100.0 % |
| Liver | Qwen | 4-option | 25 % | 165 | 10.3 | 10.3 | +0.0 | — |

The two probe kinds carry different chance levels (50 % and 25 %) and are not pooled; an earlier version of this table pooled them and reported a single accuracy against a single chance level, which is meaningless.

The four-option family (*"if this lesion grew uniformly, which structure would it
reach first?"*, one correct target against three distractors within a 6× gap
ratio) is the corpus's second probe kind. Both models score at or below its 25 %
chance level in seven of eight cells — M3D-LaMed reaches 10.4 % on colon, Qwen
10.3 % on liver — and the image contributes −8.9 to +4.4 pp. We do not build on
these numbers: the family is small (45–165 items per organ), it has no matched-pair
structure, and its distractor construction has not been audited for the leakage
we found in the binary family (§7.6). It is reported for completeness.

M3D-LaMed's colon binary cell is 53.2 % (400/752, ≈1.8σ). This is a full-corpus
figure and therefore carries the growth-magnitude cue; on the growth-matched
subset the same model is 50.1 % (§7.4, §7.6). Where this paper says "at chance"
it means on confound-free items.

Qwen2.5-VL answers both members of a matched pair identically in **100.0 % of
4,238 pairs**. Since the golds are opposite, such a pair scores exactly 50 % by
arithmetic: the model's chance-level accuracy is not a statistical coincidence
but a consequence of its own inconsistency.

### 7.5 Grounding: the ROI arms

The per-organ VGR values for all four models, the arms as MedVIGIL defines
them, are in Appendix A.6. They are superseded by the decomposition below,
which uses the same runs plus two arms the definition does not include.

**A second architecture reproduces the sign, and a four-arm decomposition says
what the sign means.** Qwen2.5-VL-32B — 7/7 on the response controls, 50.8 %
modal share, the model §7.4 identifies as load-bearing — was run on the same
items with `--fill local` passed explicitly, and on two further arms the VGR
definition does not use: `full`, the untouched volume, and `zero`, a blank one.

| Organ | full | roi_masked | roi_only | zero | VGR | mask cost | rest cost | signal |
|---|---|---|---|---|---|---|---|---|
| Lung (n = 394) | 58.4 % | 56.6 % | 50.3 % | 50.0 % | **−6.3** | +1.8 | **+8.1** | +8.4 |
| Colon (n = 600) | 56.8 % | 57.5 % | 52.5 % | 50.0 % | **−5.0** | −0.7 | **+4.3** | +6.8 |
| Pancreas (n = 600) | 56.2 % | 56.3 % | 50.2 % | 50.0 % | **−6.2** | −0.2 | **+6.0** | +6.2 |
| Liver (n = 600) | 54.3 % | 54.3 % | 49.8 % | 50.0 % | **−4.5** | +0.0 | **+4.5** | +4.3 |

And InternVL3-8B, a different family at a quarter the size, on the same items:

| Organ | full | roi_masked | roi_only | zero | VGR | mask cost | rest cost | signal |
|---|---|---|---|---|---|---|---|---|
| Lung (n = 394) | 60.4 % | 60.4 % | 53.8 % | 50.0 % | **−6.6** | +0.0 | **+6.6** | +10.4 |
| Colon (n = 600) | 58.7 % | 60.8 % | 52.5 % | 50.0 % | **−8.3** | −2.1 | **+6.2** | +8.7 |
| Pancreas (n = 600) | 54.8 % | 58.3 % | 52.7 % | 50.0 % | **−5.7** | −3.5 | **+2.2** | +4.8 |
| Liver (n = 600) | 57.7 % | 58.2 % | 52.5 % | 50.0 % | **−5.7** | −0.5 | **+5.2** | +7.7 |

*All four organs carry the explicit-`local` re-runs. Lung and Colon reproduce
the original unattested local-fill measurement to the decimal (−6.6 and −8.3);
Pancreas does not, which is why the other two were re-run rather than assumed.*

> **These four rows are being re-run with the fill on the command line.** Their
> masked arms predate the per-row `fill` field, so the analysis refuses to
> reproduce them: for three organs neither arm records a fill and for Liver one
> arm records `local` against one that does not. The values are almost certainly
> `local` — every run post-dates the default change — but "almost certainly" is
> what the field exists to replace. The `full` and `zero` arms are unaffected,
> so the mask-cost and rest-cost columns are the ones under re-measurement, not
> the signal column.

*mask cost* = full − roi_masked, what removing the evidence costs. *rest cost* =
full − roi_only, what removing everything else costs. *signal* = full − zero,
what the volume is worth at all.

And Qwen3-VL-8B, a third family:

| Organ | full | roi_masked | roi_only | zero | VGR | mask cost | rest cost | signal |
|---|---|---|---|---|---|---|---|---|
| Lung (n = 394) | 53.3 % | 54.3 % | 54.6 % | 50.0 % | +0.3 | −1.0 | −1.3 | +3.3 |
| Colon (n = 600) | 54.8 % | 56.0 % | 56.5 % | 50.0 % | +0.5 | −1.2 | −1.7 | +4.8 |
| Pancreas (n = 600) | 56.3 % | 55.5 % | 54.2 % | 50.0 % | −1.3 | +0.8 | +2.2 | +6.3 |
| Liver (n = 600) | 55.0 % | 55.5 % | 52.5 % | 50.0 % | −3.0 | −0.5 | +2.5 | +5.0 |

And InternVL3-14B, the fourth family and the largest montage model after
Qwen-32B:

| Organ | full | roi_masked | roi_only | zero | VGR | mask cost | rest cost | signal |
|---|---|---|---|---|---|---|---|---|
| Lung (n = 394) | 53.0 % | 51.3 % | 49.2 % | 50.0 % | −2.0 | +1.8 | +3.8 | +3.0 |
| Colon (n = 600) | 54.2 % | 53.5 % | 50.5 % | 50.4 % | −3.0 | +0.7 | +3.7 | +3.8 |
| Pancreas (n = 600) | 53.2 % | 51.7 % | 50.3 % | 50.0 % | −1.3 | +1.5 | +2.8 | +3.2 |
| Liver (n = 600) | 52.8 % | 52.5 % | 51.0 % | 50.3 % | −1.5 | +0.3 | +1.8 | +2.5 |

**Removing the evidence region costs nothing. That is the invariant, and it is
the one the claim needs.** Across sixteen organ-model cells in four families the
mask cost runs from −3.5 to +1.8 pp: zero to within noise in every cell, and
negative in seven of them. The `zero` arm sits at 50.0–50.4 % throughout, so
the volume is worth 2.5 to 10.4 points and **none of that worth survives the
loss of the evidence region, because none of it was there.**

**What removing everything *else* costs is model-dependent, and we report that
rather than average it.** For Qwen2.5-VL-32B and InternVL3-8B the rest cost is
+2.2 to +8.1 pp and accounts for essentially the whole signal. For Qwen3-VL-8B
it is −1.7 to +2.5: on two organs the model does *better* with 99 % of the
volume removed than with all of it. A two-model version of this table would have
supported "removing everything else costs everything the volume was worth"; the
third family shows that is a property of those two models, not of the task. The
invariant survives the third family; the corollary does not. The `zero` arm lands on exactly 50.0 % in all
eight, so the volume is not irrelevant — it is worth 4.3 to 10.4 points — but
**almost none of that worth is located in the lesion, the target, or the
corridor between them**, in two families four times the size apart.

One cell had appeared to qualify the "almost", and re-measurement removed it.
Under the unattested arms, Pancreas/InternVL3 showed a mask cost of +0.7 pp and
a rest cost of +1.5 — a pattern we described as information held redundantly
inside and outside the region. Re-run with the fill on the command line, the
same cell gives −3.5 and +2.2, in line with the other five. The redundancy was a
property of the air-filled masked arm, not of the model. We record this because
it is the second time in this section that a fill artefact produced a plausible
mechanism, and the first time we had already written the mechanism down.

This resolves the negative VGR without needing the evidence region to be harmful.
VGR subtracts an arm that replaces ~99 % of the volume from an arm that replaces
~1 % of it. When the evidence contributes nothing and the remaining 99 % carries
several points, that subtraction is negative by construction. **VGR as defined
cannot separate grounding from intactness**, and in 3D the two come apart:

- `roi_only` collapses to chance (50.3, 52.5, 50.2, 49.8 %) and its response
  distribution collapses with it — 93.1 % modal on Lung against 57.1 % for the
  full volume, close to the 100 % constant the blank `zero` arm produces. Only
  12.9 % of answers change when the evidence is masked; 40.6 % change when
  everything else is.
- The same subtraction in 2D runs the other way. Every capable model there has
  *positive* VGR (+3.6…+49.5), meaning it does better on the ROI-only arm
  despite that arm having less image. An effect that survives the intactness
  asymmetry is stronger evidence than one that merely reflects it; a sign flip
  between modalities is what happens when the evidence region stops being where
  the answer lives.

What we can rule out: it is not the growth cue (VGR is a difference between arms
over identical items, so the text cue cancels exactly); it is not a broken model
(both models with a negative sign answer 7/7 on the response-channel controls);
and it is not one architecture's quirk (two families, 8B and 32B, agree on sign
and magnitude). What is not yet ruled out is the fill — the control exists but
its provenance does not — and the intactness asymmetry between the two arms.
Both are being measured, and neither is a question we can answer by argument.

What remains open beyond those is whether the evidence region carries
information these models are actively hurt by, or whether our ROI — lesion,
target and a 15 mm corridor — is simply not the region they use. Separating
those means changing how the region is constructed, which we have not done.

We state the negative result, the controls that are pending and what each would
show, rather than a mechanism we cannot support. With them landed, the
volumetric Grounding axis rests on two measurable models that agree on a
negative sign in every organ, two that cannot be measured at all, and a
decomposition showing the sign is a property of the metric's arms rather than of
the evidence region. We do not average that into a single Ground score.

### 7.6 A text-only cue in the generated corpus

We found a language-accessible cue in our own corpus and report it in full,
because it bears directly on assumption **A2** and because every conclusion in
this paper is restated on the subset that removes it.

**The cue.** Matched pairs straddle the true gap, so the larger growth amount is
necessarily the "yes" member — within a pair, the design working as intended.
Across the corpus the *marginal* distributions differ, and a threshold on the
growth number alone, with no image and no anatomy, reaches **69.2 %**. "Chance
is exactly 50 %" bounds the label distribution inside a pair; it does not bound
what a model can read off the number across the corpus.

**The correction.** The growth-matched subset balances yes/no within 2 mm growth
bins (n = 1,368; 684/684), on which the same threshold rule falls to 50.8 %.
Because the correction selects over already-scored items, every model is
re-scored without re-running anything.

**What it cost.** Three findings depended on the confounded corpus and are
withdrawn: a capability-axis amplification gain (§7.8), a matched-pair
consistency measure equivalent to comparing two integers (§7.8), and the
apparent competence of the three models scoring 53–58 % on the full corpus,
all of which fall to chance on the subset. Appendix B gives the cue's
derivation, the per-organ threshold ceilings, the audit of which results it
touches, and the reasoning that let it survive our first review of the corpus.

### 7.7 Safety, and why the scoring rule decides it

Five families over 3,200 probes, both models, raw versus calibrated — **identical
forward passes, only the scoring rule differs.**

| | M3D raw | M3D calib. | Qwen raw | Qwen calib. |
|---|---|---|---|---|
| Acc_orig | 0.0 % | 45.5 % | 0.5 % | 39.5 % |
| PR | 0.0 % | 44.0 % | 1.0 % | 40.7 % |
| NEG | 0.0 % | 40.8 % | 1.0 % | 37.8 % |
| SDR | 0.0 % | 43.8 % | 0.2 % | 24.8 % |
| **trap refused** | **100.0 %** | **2.0 %** | **100.0 %** | **76.7 %** |
| refusal on answerable | 85.8 % | 2.8 % | 97.9 % | 17.5 % |
| **SFR$_w$** | **0.0 %** | **97.3 %** | **0.0 %** | **23.3 %** |
| committed subset (chance 50) | — | 49.3 % | — | 50.4 % |
| **MCS** | **0.0** | **7.1** | 2.0 | **49.1** |

Raw scoring reports both models as perfectly safe and wholly incapable; both
readings are artefacts of a near-constant response, since the models select the
refusal *string* on 85.8 % and 97.9 % of **answerable** probes. Calibrated, they
separate sharply: Qwen declines 76.7 % of unsatisfiable premises, spread evenly
across risk tiers (L1 28 %, L5 25 %) — genuine refusal behaviour — while
M3D-LaMed declines 2.0 %, fabricating on 97.3 % weighted.

**The scoring rule decides the composite and inverts which axis bottlenecks each
model.** Uncalibrated, both are pinned at the floor (M3D 0.0, Qwen 2.0) with
Safe = 100 for both: the refusal string dominates the likelihood and Cap
collapses. Calibration separates them by 42 points (7.1 against 49.1) and swaps
the bottleneck — M3D's Safe falls from 100 to 2.7 while its Cap rises from 0.0 to
43.5. On the Capability axis the ordering does reverse: raw M3D 0.0 < Qwen 0.7,
calibrated M3D 43.5 > Qwen 35.7. Any evaluation of refusal behaviour by likelihood
over heterogeneous option strings inherits this, including 2D work using the same
protocol. Ground inputs for these composites: M3D 50.5, Qwen 50.0 (§7.5).

The family that carries this result was built three times; the two rejected
constructions and the defects that rejected them are in Appendix A.1.

**The scoring artefact is exact and constructional**: raw likelihood scoring
reports SFR$_w$ = 0.0 % for every model on every family, because the refusal
string dominates regardless of content. Four models, two input routes, eight
arms — all 0.0 %, and three of the four fabricate on 84–99 % of the same probes
once the string prior is removed.

**The one positive safety result does not survive a better-powered version of
its own construction, and we withdraw it.** v3 and v4 are the same family —
premises false by surgical absence, no crop-prone targets — differing only in how
many probes each qualifying scan contributes. v3 draws one, which makes each
patient's rate a single coin flip; v4 draws a median of eight, over the same 81
scans, so the bootstrap's independent-unit count is unchanged while each unit's
estimate stops being a knife-edge readout.

| Image contribution (pp) | v1: 600/600 | v2: 226/226 | v3: 77/77 | **v4: 600/81** |
|---|---|---|---|---|
| Qwen2.5-VL-7B | **+22.4** *** | +0.6 n.s. | **+18.8** ** | **+5.0** n.s. |
| M3D-LaMed-Phi3-4B | **+15.1** *** | −4.7 n.s. | +7.9 n.s. | +4.4 n.s. |
| InternVL3-8B | — | +1.7 n.s. | −5.3 n.s. | **−4.8** * |
| Qwen2.5-VL-32B | — | **−32.5** *** | **−48.2** *** | **−52.0** *** |
| Qwen2.5-VL-7B, raw margin (nats) | +0.417 | +0.161 | +0.012 | +0.025 |
| Qwen2.5-VL-7B, decisions near a tie | 60 % | 58 % | 62 % | 57 % |

*probes/scans; \*\*\* $p<.001$, \*\* $p<.01$, n.s. $p>.05$.*

On v3, Qwen2.5-VL-7B's 18.8 pp rests on fourteen decisions flipping between arms
out of seventy-seven, net six, while 62 % of its calibrated decisions sit within
half a nat of a tie and its raw refusal margin moves by 0.012 nats. On v4 the
same measurement has 56 flips over 36 scans, net 18 — no longer knife-edge — and
lands at **+5.0 pp [−1.4, +11.4]**. The interval excludes an effect the size of
v1's. Neither model shows a significant image contribution on the clean
construction once each patient is measured properly.

This is the third time in this audit that a positive result on the safety axis
turned out to depend on how it was scored rather than on what the model did, and
the diagnostics that caught it — the raw margin and the near-tie share — are
reported beside every rate for that reason.

**The one effect that strengthens as the construction gets cleaner runs the
wrong way.** Qwen2.5-VL-32B — 7/7 on the response controls, the most balanced
response distribution in the audit — fabricates *more* when it can see the scan,
and the effect grows from −32.5 pp on v2 to −48.2 on v3 to **−52.0 pp** on v4
[−64.5, −40.5], $p$ = .0005. It is the only safety-axis result that agrees
across both scorings: its uncalibrated refusal margin drops by 0.822 nats when
the volume is shown, so the finding does not depend on the calibration at all.
InternVL3-8B shows the same sign at a fraction of the size (−4.8 pp, $p$ = .029;
margin −0.077). Whatever these models extract from the volume on an
unsatisfiable premise, it argues against declining.

**Tier profiles are not reported.** v3's census is L3 55 / L4 20 and contains no
L1, L2 or L5 items at all; v2's L5 cell was a single heart probe, and it contains
no aorta traps (the aorta is segmented in every scan, so it is never absent) and
no spinal-cord traps (excluded as unreliably segmented). Any risk gradient read
off these cells would reflect which organs happen to be resectable. Supplying one
would need a construction that can produce absent L1 and L5 structures, which
this one cannot.

**Refusal is grounded in one model and not the other.****Refusal is grounded in one model and not the other.**

Measured on the **v3 resectable-organ** family (6 structures, n = 75 per arm),
which is the family we report:

| | M3D-LaMed | | Qwen2.5-VL-7B | |
|---|---|---|---|---|
| | sighted | blind | sighted | blind |
| SFR$_w$ (v4, clean construction) | 96.4 % | 91.0 % | 43.7 % | 64.5 % |
| image contribution | +4.4 pp [−1.3, +10.5] | | +5.0 pp [−1.4, +11.4] | |
| amplified (CV-selected $\alpha$) | **79.4 %** | | 44.6 % | |
| amplification effect | **−17.4 pp** [−30.5, −9.1] | | −1.5 pp [−5.0, +7.0] | |

**Neither model's silent-failure rate is significantly reduced by the volume at
the decision level, and one model's is reduced substantially once the visual
contribution is amplified.** The distinction matters: an image contribution of
+4.4 pp that does not clear significance and a 17.4 pp reduction from scaling the
sighted-versus-blind difference are consistent, and together they say the signal
is in the logits and out-voted rather than absent. A study reporting only the
capability axis would have concluded M3D-LaMed cannot use a CT volume. It can —
just not enough to change what it answers. M3D-LaMed shows
the opposite sign (−6.6 pp), separating the architectures again.

**The single-phrase family overstated this threefold** (+45.6 pp), and the
verified-absent number is the one we report. The image moves refusal by +28 pp at
L1 (n = 67), +30 at L3 (n = 144) and +33 at L4 (n = 12); the L2 and L5 cells
(n = 2 and n = 1) are not interpretable and we print no value for them. The
unweighted contrast over all 226 items is larger and tighter than the weighted
headline: **39.4 % sighted against 69.0 % blind, +29.6 pp [+23.5, +35.8]**.

### 7.8 Matched-pair consistency

Two quantities are reported, and keeping them apart is the whole point. The
**differentiation rate** is how often a model gives the pair two different
answers at all; the **direction accuracy** is, among those pairs, how often the
"yes" landed on the nearer structure. Only the second is a grounding measure,
and its null is exactly 50 % by construction — independent of how often the
model chooses to differentiate.

| Target-contrast pairs (600 pairs, 1,200 probes per arm) | responses | differentiates | direction acc. | 95 % CI | $p$ | Holm $p$ |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-32B — sighted | 646 yes / 554 no | 49.0 % | 166/294 = **56.5 %** | [50.7, 62.0] | .031 | **.092** |
| InternVL3-8B — sighted | 384 yes / 816 no | 42.0 % | 134/252 = **53.2 %** | [47.0, 59.2] | .345 | **.345** |
| M3D-LaMed-Phi3-4B — sighted | 201 yes / 999 no | 20.8 % | 74/125 = **59.2 %** | [50.4, 67.4] | .049 | **.097** |
| Qwen2.5-VL-7B — sighted | 1,200 no | 0.0 % | n/a | | | |
| *all four* — blind | ≥ 1,199/1,200 one token | ≤ 0.2 % | n/a | | | |

**The image moves the response distribution and does not carry the spatial
relation.** Blind, every model answers with a single constant token and
differentiates essentially no pairs. Shown the volume, three of the four begin
to differentiate — Qwen-32B on half the pairs, InternVL3 on 42 %, M3D on 21 % —
so the image is demonstrably being used for *something*. But among the pairs
each model does split, the "yes" lands on the nearer structure 56.5 %, 53.2 %
and 59.2 % of the time, and after Holm correction across the three tests **no
model is distinguishable from chance direction** ($p$ = .092, .345, .097).

We flag one methodological trap here because we fell into it. The natural
score for these pairs — answered differently *and* both correct — has a chance
level of 25 %, not 0, since two coin flips match two opposite golds a quarter of
the time. Scoring it against the blind arm's 0 % appears to show large image
gains (+27.7, +22.3, +12.3 pp), but the blind 0 % is produced by constant
answering, not by an absent anatomical prior; the comparison credits the image
for nothing more than moving a model off a fixed token. Under that inflated
reading, all three effects clear $p<0.001$. Under the correct null, none of them
clear .05. Any matched-pair design whose blind arm can go degenerate needs the
null stated in the metric rather than borrowed from a control arm.

This is the measurement the growth-contrast pairing could not provide, and it
resolves the question that pairing left open. Matched-pair consistency reflects
the *presence* of the image — reliably, in three architectures, from a constant
blind baseline. It does not, in any of them, reflect the geometry the question
asks about. That is a narrower claim than the one we set out to test and a
sharper one for the argument of this paper: a benchmark could report these
models as strongly image-dependent, on a metric that is real, while the spatial
relation it purports to measure is absent.

**An amplification intervention on the capability axis was withdrawn.** Scaling
the sighted-versus-blind logit difference raised confound-free accuracy by up to
+7.3 pp with out-of-fold $\alpha$ selection and patient-level bootstrap — and the
gain was confined to items carrying the growth cue of §7.6. On the matched
subset it is +0.0. Appendix A.2 gives the full sequence, including a re-run with
3.9× the data that reproduced the effect before the confound was found; every
guard conventional in this literature passed it.


**The intervention works — on the axis where the image carries signal.** The test
above was run on the capability axis, where §7.7 subsequently showed the image
contributes +0.0 pp. Testing an amplifier on a channel with no signal and
concluding the amplifier fails is a design error, so we repeated it on the safety
axis, where the trap probes carry no growth–label relation at all (§7.7):

| Amplification on trap probes | SFR$_w$ at $\alpha{=}1$ | at CV-selected $\alpha$ | change | 95 % CI |
|---|---|---|---|---|
| **M3D-LaMed-Phi3-4B** (v4, 600 probes / 81 scans) | 96.8 % | **79.4 %** | **−17.4 pp** | **[−30.5, −9.1]** |
| Qwen2.5-VL-7B (v4, 600 / 81) | 46.1 % | 44.6 % | −1.5 pp | [−5.0, +7.0] |
| M3D-LaMed-Phi3-4B (v1, 600 / 600) | 97.6 % | 87.6 % | −10.0 pp | [−13.1, −5.8] |
| Qwen2.5-VL-7B (v1, 600 / 600) | 23.0 % | 3.3 % | −19.7 pp | [−25.7, −15.1] |

**Amplification survives the move to the clean construction for the model that
fabricates, and does not for the model that does not.** On v4 — verified
surgical absence, no crop-prone targets, eight probes per scan — scaling
M3D-LaMed's sighted-versus-blind logit difference cuts its weighted
silent-failure rate by **17.4 pp**, a larger effect than on the weaker v1 family
and still significant with 81 independent scans rather than 600. Qwen2.5-VL-7B's
19.7 pp on v1 does not replicate on v4 (−1.5 pp, [−5.0, +7.0]), the same pattern
its image contribution shows in §7.7: an effect that lives on the family whose
premises are assumed false rather than verified.

That split is the result, and it is not the one we expected. The model whose
refusal behaviour is *worst* — M3D-LaMed fabricates on 96.4 % of these probes and
its response channel answers 2 of 7 control questions — is the one carrying
recoverable visual signal in its logits. The model with a clean response channel
carries none that amplification can reach.

And the mechanism is the one the capability axis could not show: M3D's *raw*
sighted-versus-blind SFR difference on v3 is +7.9 pp and not significant,
because its decision is prior-dominated — but scaling the sighted-blind
difference turns that difference into refusals. **The visual signal is present
and under-weighted**; it simply is not present on the axis we first looked at.

So the withdrawal is specific, not general: **amplification does not recover
spatial-reasoning accuracy** (§7.8 above, n = 3,074, confound-free, no cell above
chance), **and it does recover refusal behaviour** on unsatisfiable premises. An
intervention that helps a model decline what it cannot see, while not helping it
reason about what it can, is a coherent and reportable result — and it is what the
axis-by-axis image contribution predicts.

**What the earlier failure taught.** Labels
balanced by construction, patient-level rather than item-level bootstrap,
out-of-fold $\alpha$, and a corrected estimator that widened the intervals by
resampling the whole cross-validation: all held, all passed a false result, and
none could have done otherwise, because the confound lived upstream of them in the
corpus. The check that caught it is arithmetic over stored predictions.

**Where the confound stops.**

| Result | Affected | Why |
|---|---|---|
| §7.4 headline accuracies | No | all at 50 %; matching moves each ≤ 0.7 pp |
| §7.4 InternVL / Qwen-32B above chance | **Yes** | fully attributable |
| §7.5 VGR | No | a difference between arms over identical items |
| §7.7 SFR, calibration, trap grounding | No | trap items carry no growth–label relation |
| §7.8 amplification, capability axis | **Yes — withdrawn** | gains confined to confounded items |
| §7.8 amplification, safety axis | No | trap items carry no growth–label relation; −19.7 / −10.0 pp hold |
| §7.8 consistency-as-grounding (growth pairs) | **Yes — withdrawn** | identical to ordering by the number |
| §7.8 target-contrast pairs | No | both members carry the same growth amount by construction |
| §7.3 response-channel controls | No | separate question set, no growth parameter |

### 7.9 Language-prior accuracy

| | Sighted | Blind | Volumetric task (matched) |
|---|---|---|---|
| M3D-LaMed-Phi3-4B | **55.0 %** | **61.5 %** | 50.1 % |
| Qwen2.5-VL-7B | **64.5 %** | **62.0 %** | 50.0 % |

Against a 20 % chance level both models hold the anatomy, and M3D-LaMed scores
**100 %** on *"if two structures are separated by 30 mm, would growth of 5 mm
suffice for contact?"* — the benchmark's own reasoning stated abstractly — while
sitting at 50 % when the same reasoning must be read off a volume. Neither model
is helped by the image on this axis, as an image-independent axis should behave.

One inconsistency we report rather than explain: the *qualitative* form of the
same question — "does isotropic growth increase the likelihood of contact?" —
scores **5 %**, while the quantitative form scores 100 %. We can say the reasoning
is available in concrete numeric form; not that the model holds the general
principle.

**Earlier LPA figures are withdrawn**: the first 3D knowledge bank emitted one
question 200 times with options reshuffled, which measures option-order
sensitivity rather than knowledge breadth and yields non-independent items.

### 7.10 Does the volume reach the decision?

Chance-level accuracy is compatible with two states that no accuracy number can
separate: nothing usable reaches the decision, or something does and is
out-voted. The separation is visible in the gap between how far the volume moves
the scores and how often it moves an answer. Over the paired runs on identical
probes, per model: **perturbation** = mean $|$sighted $-$ blind$|$ across the
option logprobs; **decision gap** = mean $|$top1 $-$ top2$|$ in the sighted arm,
the distance a perturbation must cover to change anything; **flip rate** = share
of probes whose argmax differs between arms, bootstrapped over volumes.

| Model | Input | Perturbation | Decision gap | Flip rate | 95 % CI |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | montage | 0.355 | 1.657 | **0.0 %** | [0.0, 0.0] |
| Idefics3-8B | montage | 0.674 | 7.158 | **0.0 %** | [0.0, 0.0] |
| Med3DVLM-7B | native | 0.821 | 13.599 | **0.0 %** | [0.0, 0.0] |
| SmolVLM2-2.2B | montage | 0.264 | 0.576 | 0.1 % | [0.0, 0.2] |
| Qwen2.5-VL-3B | montage | 0.355 | 0.206 | 0.5 % | [0.2, 0.8] |
| LLaVA-OneVision-7B | montage | 0.109 | 0.896 | 1.2 % | [0.6, 1.9] |
| M3D-LaMed-Llama2-7B | native | 4.626 | 2.923 | 18.5 % | [15.6, 21.7] |
| M3D-LaMed-Phi3-4B | native | 6.774 | 8.565 | 19.7 % | [17.0, 22.4] |
| InternVL3-8B | montage | 0.905 | 0.938 | 30.3 % | [27.5, 33.0] |
| Qwen2.5-VL-32B | montage | 1.359 | 0.964 | **48.9 %** | [45.5, 52.2] |
| Pixtral-12B | montage | 1.011 | 0.437 | **57.3 %** | [54.1, 60.3] |
| Qwen3-VL-8B | montage | 2.319 | 1.985 | **77.4 %** | [74.8, 79.9] |
| InternVL3-14B | montage | 2.345 | 0.806 | **92.0 %** | [90.7, 93.4] |

> **Figure 10.** Perturbation against flip rate, one point per model, montage and
> native in different markers, with the line perturbation = decision gap drawn.
> Data: `figdata/fig10_margin.csv`.

**The chance-level result has two mechanisms behind it, and they are not the
same failure.** Six models perturb their scores measurably and change almost
nothing: Qwen2.5-VL-7B moves 0.355 nats against a 1.657-nat decision gap and
flips 0.0 % of 2,262 probes; Med3DVLM-7B moves 1.248 nats against a 14.460-nat
gap. For these the volume arrives and is out-voted, which is the state the
amplification intervention of §7.8 is built for.

**Five models are the opposite case, and it is the harder one to explain away.**
InternVL3-14B changes its answer on **92.0 %** of probes when shown the volume,
Qwen3-VL-8B on 77.4 %, Pixtral-12B on 57.3 %, Qwen2.5-VL-32B on 48.9 %,
InternVL3-8B on 30.3 %.
These are not models ignoring an image. The volume reaches the decision and
rewrites most of it — and confound-free accuracy is 50.1 %, 48.7 %, 48.0 % and
50.6 % respectively, every interval containing 50. **The image changes the
answers without making them right.**

That distinction matters for what a remedy should target. Amplifying an
under-weighted signal is the right instrument for the first group and cannot
help the second: there is nothing under-weighted about a signal that already
rewrites nine answers in ten. For those models the content of what arrives, not
its weight, is what fails.

We report no 2D counterpart to this table. The published 2D audit provides
per-model metrics rather than logits, so the same quantities cannot be computed
there, and a cross-modality comparison of them would not be reproducible.

### 7.11 Radiologist validation **[PENDING]**

104 probes, stratified over organ × severity tier × margin difficulty, exported
with lesion and target outlined on a slice through the lesion centroid; answer key
ships separately and the reader form carries no label column.

Two constraints are enforced by assertion at export time, because both were
violated by a first build and neither is recoverable once a reader has begun. **No
matched pair appears in the form** — members differ only in growth amount and
always have opposite golds, so showing both makes the second answerable without
the image; the first build contained one such pair, listed adjacently. **Answers
are exactly balanced** (52/52); the first build was 73 no / 32 yes, against which
a reader answering "no" throughout would have scored 69.5 %.

Coverage: liver 35, lung 25, pancreas 24, colon 20; L1 15, L2 20, L3 21, L4 24,
L5 24. To be reported: human–geometry agreement and Cohen's κ overall and per
stratum (the tight-margin stratum, |growth − gap| ≤ 3 mm, most informative);
clinical relevance (1–5) per family and tier; and severity-tier concordance with
the target-structure mapping used for SFR weighting. This is the study's answer
to **A1**.

---

## 8. Cross-Modality Analysis

The two audits share a framework, a metric set and a composite, so the
differences between them are differences in what the modality affords rather
than in how it was measured.

### 8.1 The Grounding axis collapses

| | 2D | 3D |
|---|---|---|
| VGR range (capable models) | **+3.6 … +49.5** | **−8.3 … +0.8** |
| Sign | positive for every model | negative in every organ for both models with a verified response channel |
| Discriminating? | yes, 46-point range | no; never above +0.8 |
| Exactly +0.0 cells | none | **4 of 4** (Qwen2.5-VL-7B, all organs) |
| What the arms differ in | evidence region | evidence region *and* how much image survives (§7.5) |
| Radiologist VGR | +4.5 | — |

In 2D the axis separates models. In 3D it is pinned, and for one model
arithmetically so: a system answering both members of every matched pair alike
cannot respond to any image manipulation whatsoever. A Ground axis at its floor
still contributes ~50 to a harmonic mean and reads as partial competence. **We
recommend that in the volumetric regime the Ground axis gate the composite rather
than be averaged into it**: a system whose pair-violation rate exceeds a threshold
should have its MCS reported as undefined.

### 8.2 Visual information decay: L\* = 0 in the volumetric regime

The identical protocol — Gaussian blur at $\sigma\in\{0,2,4,8,16,32,64\}$ plus a
no-image control, $L^*$ the smallest $\sigma$ at which residual visual
contribution falls below 20 % of the clean-image contribution — run on the
growth-matched volumetric corpus. Blur is applied in-plane only, with $\sigma$ in
in-plane voxels, so anisotropic slice spacing does not turn the degradation into
a different physical quantity per scan.

| Modality | Model | $\sigma$=0 | no-image | clean contribution | $L^*$ |
|---|---|---|---|---|---|
| 2D | GPT-4o | 59.0 % | 5.3 % | **+53.7** | **64** |
| 2D | Claude Sonnet-4.6 | 65.7 % | 9.3 % | **+56.3** | 32 |
| 2D | HuatuoGPT-V-7B | 55.3 % | 20.7 % | **+34.7** | 32 |
| 2D | Qwen3.5-397B | 50.3 % | 10.0 % | **+40.3** | 16 |
| **3D** | **M3D-LaMed-Phi3-4B** | **52.8 %** | **50.0 %** | **+2.8** | **8** |
| **3D** | **Qwen2.5-VL-7B** | **50.0 %** | **50.0 %** | **+0.0** | **0** |

Under one protocol and one criterion, **the language-takeover point falls by an
order of magnitude between modalities (16–64 against 0–8) and the clean-image
contribution by one to two orders (34.7–56.3 pp against 0.0–2.8 pp).**

The two volumetric models fail differently, and the difference is worth stating.
M3D-LaMed is *not* at zero: a clean image is worth +2.8 pp to it, blurring to
$\sigma=8$ removes that, and 83 of 400 probes change answer somewhere along the
sweep. It uses the image, barely, and the ratio test therefore returns a
well-defined $L^*=8$. Qwen2.5-VL is the degenerate case — 50.0 % at every level
including no-image, **zero** of 249 predictions changing anywhere in the sweep —
so its $L^*$ is reported as 0 with the denominator's collapse stated rather than
hidden.

**Qwen's flat curve is not an insensitive model.** Across those 249 probes the
option log-likelihoods move by a median of 0.53 nats between clean image and no
image, and 1.06 between clean and $\sigma=64$: the image perturbs the
distribution and never reaches the decision. That is one of the two mechanisms
§7.10 separates -- and it holds for six of the thirteen models, not all of them --
seen through the
conference version's own ablation, and it is why a single flat line is a result
rather than a missing measurement.

### 8.3 Language priors are high in both, and mean different things

| | 2D | 3D |
|---|---|---|
| Model LPA | 91.9–99.6 % | 55.0 / 64.5 % |
| Radiologist | 93.6 % | — |
| Chance | 20 % | 20 % |

Levels are **not** comparable — different banks, different question difficulty —
and we do not compare them. What transfers is the *role* of the axis. In 2D, high
LPA established that models hold the knowledge, so failures elsewhere are not
knowledge deficits. In 3D the same axis does more work: both models hold the
anatomy, one solves the benchmark's own reasoning stated abstractly at 100 %, and
both sit at chance when that reasoning must be read off a volume. **The 3D
failure is located precisely between knowledge and perception, and the LPA axis is
what locates it.**

### 8.4 Safety, and a measurement artefact that only 3D exposed

2D SFR ranges 22.0–54.3 % against a radiologist's 5.8 %. On the cleanest
volumetric trap family, calibrated SFR is 43.7 % (Qwen2.5-VL-7B), 71.5 %
(InternVL3-8B), 73.2 % (Qwen2.5-VL-32B) and 96.4 % (M3D-LaMed) — the first
inside the 2D range, the rest above all of it, and the highest belonging to the
only purpose-built medical volumetric model in the set.

The 3D audit exposed two scoring artefacts, and both apply to any suite using
the same format. **Likelihood over options of unequal length awards a perfect
Safe axis to a near-constant refusal** — every model here scores exactly 0.0 %
weighted silent-failure before calibration, which collapses a 42-point composite
difference between two models that behave nothing alike. **And the calibration
that fixes it introduces a second artefact if applied per arm**: a model's
content-free answer is not the same with and without an image, so scoring each
arm against its own baseline makes their difference partly a difference of
baselines (§7.7, Appendix A.4). The 2D suite scores the same five-option format
and reports the same kind of sighted-blind difference. We recommend content-free
calibration with a common baseline there as well, and note that we cannot say
from outside whether the published 2D values are affected.

### 8.5 What the volumetric regime adds to the framework

Two additions were forced by 3D and we believe both generalise. The
**response-channel control** — seven known-answer questions — was unnecessary in
2D, where all audited models answer them trivially, and is indispensable in 3D,
where neither native volumetric system passes. And the **image-contribution split
by axis**: the same model, same volumes, same forward passes, yields +0.0 pp on
Capability and a 17.4 pp recoverable reduction in silent failure on Safety. A
single "does the model use the image" verdict is not well-posed; the answer
depends on which axis is asked, and on whether the question is what the model
answers or what its logits contain.

---

## 9. Discussion

### 9.1 What the two modalities say together

The 2D audit found models that are visually anchored but unsafe: VGR positive
for every model in its headline table (+3.6…+49.5), SFR 22–63 % against a radiologist's 5.8 %, a composite reaching 69.2 against the reader's 83.3. The volumetric audit finds something different
in kind. Capability is at chance, Grounding is at its floor, and the composite is
not measuring a weaker version of the same thing — for M3D-LaMed it is
bottlenecked by a Safety axis at 2.7 that only calibration reveals.

The single most transferable observation is that **the axes fail independently
and must be read independently**. On the trap family built to test it,
M3D-LaMed's Safety axis carries recoverable visual signal (−17.4 pp under
amplification) while its Grounding axis cannot be measured at all and its
Capability axis moves +0.1 pp, in the same model on the same volumes. Averaging them into one number reports the mean of a channel that
responds and a channel that cannot.

**Two instruments designed for different axes turned out to measure one thing.**
The response-channel control asks whether a model can say "no" to *is this a
photograph of a cat*; the hallucination traps ask whether it can say "the
gallbladder is not in this scan" when asked whether a lesion would reach it. We
built them for different purposes — one to decide whether a model is scorable at
all, the other to score its Safety axis — and they report the same property.
Every model that fails the control fails it by polarity, near-perfect in one
answer direction and near-chance or worse in the other (§7.3), and across the five
models measured on both instruments the control's polarity orders the trap
behaviour:

| Model | Controls: correct on "no" answers | Traps: calibrated fabrication |
|---|---|---|
| M3D-LaMed-Llama2-7B | **25 %** | **100.0 %** |
| M3D-LaMed-Phi3-4B | 61 % | 96.4 % |
| InternVL3-8B | 100 % | 71.5 % |
| Qwen2.5-VL-32B | 100 % | 73.2 % |
| Qwen2.5-VL-7B | 100 % | 43.7 % |

Spearman $\rho = -0.90$ over five models — descriptive rather than a test at
this n, but the ordering is exact and the endpoints are far apart: the model
that answers "no" correctly a quarter of the time on seven trivial questions
fabricates on **every one** of 600 unsatisfiable premises. A model that cannot
produce a negative answer about the image does not produce one about the
anatomy either.

This has a practical consequence for anyone building such an audit. The
response-channel control is cheap — seven questions, twenty volumes, minutes of
compute — and it is not merely a gate on interpretability; it forecasts the
Safety axis. Running it first would have told us most of what the trap families
took days to establish.

A second observation is about the instruments rather than the models. Four of
the measurements in this audit turned out to be sensitive to a choice that looks
like a formatting detail:

- **The prompt's description of the input.** Framing every probe with "shown as
  axial, coronal and sagittal views" is true of a montage and false of a volume.
  Applied to the native models it cost M3D-LaMed-Phi3 41 of 140 control
  questions and turned M3D-LaMed-Llama2's image contribution from +0.1 into a
  significant −3.1 pp — an anomaly we would have had to explain (§7.3, §7.4).

- **The masked-region fill.** Replacing the evidence with the volume's 1st
  percentile carves an air cavity into soft tissue — a detectable artefact, not
  missing evidence (§7.5).
- **The calibration baseline.** A model's content-free answer is not the same
  with and without an image, so scoring each arm against its own baseline
  normalises them differently and their difference mixes the model's response to
  content with the drift of its own default. Qwen2.5-VL-7B's image contribution
  on one trap family is +15.1 pp under per-arm baselines and +0.6 pp under a
  common one (§7.7).
- **The proximity of calibrated decisions to a tie.** Where 60 % of a model's
  calibrated decisions sit within half a nat of a tie, the rate is a knife-edge
  readout: Qwen2.5-VL-7B's 18.8 pp is a net six items out of seventy-seven, and
  its raw refusal margin does not move at all (+0.012 nats). We report the
  margin next to every rate for this reason.

None of these is exotic. All three would pass unremarked in a table of accuracies.

### 9.2 What the ROI arms do and do not exclude

Chance-level accuracy is diagnostically ambiguous: a model may be at chance
because nothing usable reaches it, or because what reaches it is out-voted.

We previously argued that the ROI arms exclude a third explanation — that the
deficit is detection of a degraded input — because both arms carry realistic
volumes rather than an all-zero one. **The four-arm decomposition shows that
argument was wrong.** The two arms are realistic but not equally intact:
`roi_only` replaces about 99 % of the volume and `roi_masked` about 1 %. Adding
`full` and `zero` separates them, and the separation is unambiguous across four
organs — removing the evidence costs nothing (+1.8, −0.7, −0.2, +0.0 pp) while
removing everything else costs the whole 4–8 points the volume is worth (§7.5).
`roi_only` does not merely lose the rest of the image; it drives the model toward
the constant response the blank arm produces, 93.1 % modal against 57.1 % on the
full volume.

So degradation is not excluded — it is the larger of the two effects VGR
subtracts. What the decomposition does establish is stronger than what we
claimed: the volume carries several points of signal on these items, and none of
it is located in the lesion, the target, or the corridor between them.

§7.10 shows the prior-dominated picture is only half the story. Six models do
behave that way — Qwen2.5-VL-7B moves 0.355 nats against a 1.657-nat decision
gap and flips no decisions at all. But five others flip 30 to 92 % of their
answers when shown the volume and remain at chance, so for them the volume is
not out-voted; what it carries is wrong. Whether the perturbation in the first
group carries task-relevant information is the question amplification was built
to answer, and its answer is axis-dependent: on capability it does not
survive the confound (§7.8), while on safety it reduces weighted silent-failure
rate by 19.7 and 10.0 pp over 600 probes. Visual signal is present and
under-weighted on the axis where refusal is the action; on the axis where
geometry is the action we could not show it is present at all. Distinguishing
"absent" from "present but out-voted" there requires an instrument that does not
route through task accuracy — linear readout of geometric quantities from frozen
features is the obvious candidate, and we did not run it.

### 9.3 Scale is not the remedy, and the corpus shows why that is hard to see

The standard response to chance-level benchmark performance is a larger model,
and one family gives a clean test: 3B → 32B raises full-corpus accuracy 50.0 % →
56.0 % and moves confound-free accuracy not at all. What scale improved was
willingness to distinguish pair members, which here means comparing two numbers.

The uncomfortable part is that the confounded column does not look noisy or
implausible. It looks like the expected result, moving in the expected direction,
at a reportable magnitude. It was distinguishable only by an experiment that
removed the cue, and nothing in the numbers would have prompted that experiment.
We ran it because one model sat above chance and we wanted to know why — not
because we suspected the corpus.

### 9.4 What this does not license

We measure spatial reasoning under counterfactual anatomical manipulation, not
tumour biology; isotropic growth is not a growth model. A system scoring well here
would have shown it can read geometry out of a volume — a prerequisite for
clinical spatial reasoning and nothing more. None did.

---

## 10. Limitations

1. **Four of thirteen models are unmeasurable, including one we expected to be
   fine.** All three released native volumetric medical VLMs fail the
   response-channel controls, and so does Idefics3-8B, so the spatial-reasoning
   claims rest on the nine models with a working output channel — all of them
   montage-fed. The intended native-architecture confirmation did not
   materialise: the native systems turned out to be unmeasurable rather than
   confirmatory. Two further models could not be run at all
   (HuatuoGPT-Vision-7B, Aria-25B-MoE; §7.2).

2. **The volumetric interface is a montage, and we can bound but not eliminate
   that concern.** §7.3 shows the montage delivers signal that reaches the
   decision — several models rewrite most of their answers when shown it — and
   that replacing the volume with zeros costs 4.3–10.4 points. What we cannot
   show is that a *better* interface would not reveal spatial competence the
   montage hides. The three native systems are the natural test and they are
   unmeasurable, so this remains open. A study with a scorable native
   volumetric model would settle it.

3. **The confound-free subset is a subset.** All conclusions are stated on the
   1,368-item growth-matched subset rather than the full 9,484-probe corpus,
   because the corpus as generated carries a text-side cue (§7.6). The subset is
   drawn to balance yes/no within 2 mm growth bins, and its composition is
   therefore a function of one design choice we made after seeing that the cue
   existed. We report the full-corpus numbers beside it throughout so that
   choice is visible.

4. **The reader study is the only test of the assumption the whole construction
   rests on.** A1 — that a radiologist answers the contact question the same way
   the geometry does — is not discharged by computation, and §7.11 is its only
   test. Until it reports, "verifiable" means verifiable against the simulator,
   not against a clinician.
5. **A text-only shortcut existed in the generated corpus.** Growth magnitude
   correlates with the answer through the gap it is chosen relative to (§7.6). No
   model we tested exploited it on the matched subset, and all conclusions are
   restated there — but the corpus as generated did not deliver the language-prior
   immunity its construction suggests.
6. **The lung corpus is small** (62 volumes) and its intervention cells are
   underpowered.
7. **The simulator is geometric, not biophysical.** Probes measure spatial
   reasoning capability and must not be read as predicting tumour evolution.
8. **Absolute accuracy is at chance everywhere on confound-free items.** No model
   tested is usable on this task under any condition.
9. **Anatomy segmentation is automatic.** TotalSegmentator failures propagate into
   probe construction; we observed 0/588 hard failures but did not audit
   segmentation quality independently.
7. **Abdominal-thoracic CT only.** No brain, no MR — and the 2D suite's modality
   mix is not matched by the 3D corpus, so cross-modality contrasts are between
   suites, not between matched cohorts.
8. **LPA levels are not comparable across modalities** (§7.9); only the
   within-modality contrasts are.

---

## Reproducibility

**Code and data availability.** The probe generator, the thirteen model runners,
every analysis script and the manuscript's own verification suite are released
at the repository accompanying this paper. The corpora are derived entirely from
the public Medical Segmentation Decathlon and can be regenerated end to end from
the released code; the generated probes and all model predictions are released
alongside, so every number in this paper can be recomputed without re-running a
model. No new patient data were collected and no identifiable data were used.

**How the numbers in this paper are checked.** Every table is recomputed from
the prediction files by a verification script that runs as part of the build and
fails on any disagreement beyond 0.1 pp; a manuscript-consistency pass resolves
every cross-reference and refuses a withdrawn value that survives elsewhere.
This is not a courtesy: during this audit, a value hand-computed from a file
that was still being written reached the manuscript, and so did a table row
transcribed before its analysis had been read. Both were caught only when those
tables were brought under the same recomputation as the rest, which is the
argument for placing every table under it.


All data are public (Medical Segmentation Decathlon; the 2D sources as in the
conference version). Volumetric ground truth is computed and every probe ships
the geometry that produced it. Per-module self-tests cover the orientation
conventions, the refuse-when-undecidable rules, the provenance gate and the metric
definitions; they are written to fail on the specific mistakes that would be
silent otherwise. The Med3DVLM adapter, for instance, asserts that a marker placed
high in *z* survives preprocessing on the **last** axis and not the first, because
that model's $(H,W,T)$ convention is the opposite of M3D's $(D,H,W)$ and getting
it wrong feeds coronal planes to a model expecting axial ones without raising any
error.

| Artefact | Contents |
|---|---|
| Counterfactual corpus | 9,484 probes / 588 volumes / 4 organs / 4,238 matched pairs, with provenance |
| Growth-matched subset | 1,368-item confound-free evaluation set and its selection script |
| Probe-family corpus | 3,200 probes (orig / T-CF / NEG / SDR / trap × 600, plus knowledge) |
| Main model outputs | 6 models × {sighted, blind} on a common 2,262-probe subset |
| ROI arms | 16 runs over 4 conditions |
| Response-channel controls | 7 questions × 20 volumes × 6 models, full logprobs |
| Calibration terms | content-free option scores per (volume, option set) |
| Decay sweep | 7 blur levels + no-image, growth-matched subset |
| Reader-study package | 104 probes (52/52, no matched pairs), images, blind form, separate key |

Every output row stores full per-option logprobs, not just the argmax. This is
what allowed the option-string bias to be diagnosed and corrected without
re-running a single model, and why the raw and calibrated columns of §7.7 come
from identical forward passes.

**Recommended protocol for reuse.** Evaluate on the growth-matched subset, and
report what a blind model scores on the exact items being graded. Exact label
balance does not bound text-only performance — a threshold on the growth number
alone reaches 69.2 % on the full corpus — and three of our own findings had to be
withdrawn before we measured it.

---

## Appendix A. Construction histories and withdrawn results

A finding that survived three constructions is worth more than one reported from
the construction that produced it, and a reader cannot tell them apart from the
final numbers alone. These are the rejected versions, the defects that rejected
them, and the results we withdrew.

### A.1 The hallucination-trap family, built three times

**The trap family, rebuilt twice.** The 600 traps of the first build all ask one
question — "would it contact the prosthetic implant?" — so every number on this
axis rested on a single lexical item, the same non-independence for which the
knowledge bank of §7.9 was withdrawn. Rebuilding it took three attempts, and the
two failures bound what this construction can support.

*Attempt 2* named structures with zero voxels in the volume's own segmentation,
keeping those segmented in ≥ 70 % of comparable scans. 226 traps over 38
structures — but 61 % were vertebrae, sternum, hips and gluteal muscles, absent
because the acquisition stopped short of them. `vertebrae_T8` sits at frequency
0.72 and `sternum` at 0.75, so the threshold admitted exactly the class it was
meant to exclude. Asking whether a liver lesion could contact a structure outside
the field of view tests cropping, not grounding.

*Attempt 3* is the one we report. Zero voxels in a segmentation conflates three
causes — the patient lacks the structure, the scan did not cover it, the
segmenter missed it — and neither a within-scan z-position test (a structure's
position is only observable in scans that reached it) nor a bracketing test
(ribs and vertebrae interleave with organs, so a *missed* rib is still bracketed)
separates them. What does is restricting to **solid organs that are routinely
resected and reliably segmented**: gallbladder, kidney, spleen, adrenal, uterus.
For these, zero voxels inside a scan covering their neighbours is a statement
about the patient. **75 traps over 6 structures, 0 % crop-prone.**

| | v1: 1 phrase × 600 | v2: 38 structures × 226 | **v3: 6 resectable organs × 77** |
|---|---|---|---|
| premise false because | assumed | 61 % crop / miss | **surgical absence** |
| raw SFR$_w$ (all models) | 0.0 % | 0.0 % | **0.0 %** |
| M3D-LaMed, calibrated | 97.3 % | 98.7 % | **96.9 %** |
| Qwen2.5-VL-7B, calibrated | 23.3 % | 18.1 % | **45.2 %** |
| InternVL3-8B, calibrated | — | 85.4 % | **72.5 %** |
| Qwen2.5-VL-32B, calibrated | — | 84.7 % | *running* |

Image contribution on v2, both arms scored against the same content-free
baseline (see below), bootstrapped over volumes:

| Model | response controls | image contribution | 95 % CI | $p$ |
|---|---|---|---|---|
| M3D-LaMed-Phi3-4B | 2/7 | −4.7 pp | [−17.8, +2.1] | .62 |
| Qwen2.5-VL-7B | 7/7 | **+0.6 pp** | [−9.6, +12.4] | .85 |
| InternVL3-8B | 7/7 | +1.7 pp | [−1.9, +10.6] | .58 |
| Qwen2.5-VL-32B | 7/7 | **−32.5 pp** | [−66.1, −21.9] | **.0005** |



### A.2 The matched-pair intervention, withdrawn

### 7.8 Matched-pair consistency: a withdrawn intervention and a corrected null

**Consistency is necessary but is not evidence of grounding.**

| Model | Consistent pairs | Accuracy within them | Rate of "larger growth → yes" |
|---|---|---|---|
| Qwen2.5-VL-3B | 7 | 57.1 % | **57.1 %** |
| Qwen2.5-VL-7B | 0 (100 % violation) | — | — |
| M3D-LaMed-Phi3-4B | 64 | 64.1 % | **64.1 %** |
| InternVL3-8B | 233 | 90.6 % | **90.6 %** |

The last two columns are identical in every row and necessarily so: within a pair
receiving different answers, the model is right on both exactly when it assigned
"yes" to the larger growth amount. Consistency here is achieved by comparing two
integers in the prompt. What survives is the converse: an inconsistent response
is provably a coin flip, computable with no labels.

**The metric works once the pairing removes the shortcut.** We rebuilt the
pairing so that both members state the *same* growth amount on the *same* lesion
and differ only in which structure is named — one whose gap is under that
distance, one whose gap is over it (600 pairs; growth medians identical at
20.9 mm for both labels, so a threshold on the number scores exactly 50.0 %).
Ordering two identical numbers cannot produce a differentiated answer, so
consistency here requires distinguishing two structures.

### A.3 The masked-region fill, and how its provenance was lost

> **Fill provenance, being re-measured.** The two masked arms of each row must
> share a fill, and the runner records it per row — but only since partway
> through this campaign. Of the sixteen masked arms above, one records
> `local`, one pair records `local` on both sides, and the rest predate the
> field: they were almost certainly run under the `local` default, and that is
> not the same as attested. One row is worse than unattested — Liver/Qwen-32B
> pairs a recorded-`local` arm against an unrecorded one. Every masked arm is
> being re-run with the fill given on the command line and written into the
> filename; the `full` and `zero` arms are unaffected, since neither consults a
> fill. The Lung/InternVL3 pair has already returned, and it reproduces the
> original local-fill measurement exactly (53.8 % / 60.4 %, VGR −6.6), which is
> evidence the unattested numbers are right and not evidence that we may keep
> quoting them.


### A.4 The calibration baseline, and why the arms cannot use their own

**A second artefact sits inside the correction for the first.** Content-free
calibration subtracts the model's answer-string prior, measured by asking a
contentless question on the same input — but that prior is not the same in the
two arms. Shown a volume, Qwen2.5-VL-32B's content-free answer is "yes" on all
226 probes; blind, it is the refusal on all 226. Scoring each arm against its
own baseline normalises the two arms differently, so their difference mixes the
model's response to content with the movement of its own default. We therefore
use each arm's own baseline for the per-arm *rate* — the right normalisation for
"how often does this model fabricate under this input condition" — and a single
baseline, the blind one, for the *difference*, which is the only way the two
sides differ solely in their answers.

The correction is not cosmetic and does not move everything the same way:

| Image contribution (pp) | v1: 600 | v2: 226 | **v3: 77** |
|---|---|---|---|
| Qwen2.5-VL-7B, per-arm baselines | +45.7 | +15.1 | +29.6 |
| **Qwen2.5-VL-7B, common baseline** | **+22.4** *** | **+0.6** n.s. | **+18.8** ** |
| M3D-LaMed, per-arm baselines | −0.5 | −6.6 | −2.8 |
| **M3D-LaMed, common baseline** | **+15.1** *** | **−4.7** n.s. | **+7.9** n.s. |

*** $p<.001$, ** $p<.01$, n.s. $p>.05$; bootstrapped over volumes.


### A.5 Why the intermediate trap family shows no effect

**v2 is where the effect vanishes, and the construction explains it.** 61 % of
v2's targets are skeletal or limb-girdle structures that a scan may simply not
cover, and a model can decline "the sternum" on an abdominal scan from anatomy
alone. The blind arm can already refuse, so there is nothing for the image to
add: +0.6 pp, $p$ = .85. That is the predicted consequence of the defect we
built v3 to remove, and it is why we report v3 rather than v2 as the safety
result.

The absolute rates move, and the direction is informative. Qwen2.5-VL-7B's
calibrated silent-failure rate *rises* from 18.1 % on v2 to 43.7 % on v4,
because v2's traps are easy — 61 % name skeletal or limb-girdle structures, and
a model can decline "the sternum" on an abdominal scan from anatomy alone, while
a resected gallbladder cannot be told from a present one without looking. The
clean family is therefore both the better-constructed one and the harder one. We
report all four and print the intermediate versions, because a reader given only
the final 600 probes cannot judge why those 600.

### A.6 Per-organ VGR under the two-arm definition

Reported for continuity with the conference version's metric. The four-arm
decomposition in §7.5 supersedes it: these two arms differ in how much image
survives as well as in the evidence region, so their difference cannot be read
as grounding.

| Organ | Model | roi_only | roi_masked | VGR | n |
|---|---|---|---|---|---|
| Lung | M3D-Phi3 | 51.5 % | 50.8 % | +0.8 | 394 |
| Lung | Qwen-7B | 50.0 % | 50.0 % | +0.0 | 394 |
| Lung | **InternVL3-8B** | 53.8 % | 60.4 % | **−6.6** | 394 |
| Colon | M3D-Phi3 | 50.0 % | 53.0 % | -3.0 | 600 |
| Colon | Qwen-7B | 50.0 % | 50.0 % | +0.0 | 600 |
| Colon | **InternVL3-8B** | 52.5 % | 60.8 % | **−8.3** | 600 |
| Pancreas | M3D-Phi3 | 49.8 % | 51.7 % | -1.8 | 600 |
| Pancreas | Qwen-7B | 50.0 % | 50.0 % | +0.0 | 600 |
| Pancreas | **InternVL3-8B** | 52.7 % | 58.3 % | **−5.7** | 600 |
| Liver | M3D-Phi3 | 52.5 % | 52.7 % | -0.2 | 600 |
| Liver | Qwen-7B | 50.0 % | 50.0 % | +0.0 | 600 |
| Liver | **InternVL3-8B** | 52.5 % | 58.2 % | **−5.7** | 600 |

One cell in that table was re-measured. An integrity sweep over every result
file — each pair of arms entering a difference must cover identical items —
found the Colon/Qwen-7B `roi_only` arm holding 564 items against its masked
arm's 600, so the difference reported for it had been taken over two different
item sets. Re-run to 600 matched items with the fill recorded per row, it gives
+0.0, the same value the mismatched comparison gave; we re-ran it rather than
keep the number, because a difference over mismatched items is not a
measurement whatever it evaluates to. The sweep now runs as part of the
pipeline and the job runner refuses any output whose row count is short.

Both ROI arms carry realistic volumes rather than an all-zero one, so neither is
degraded in the sense a blank input would be. They are not, however, degraded
*equally*, and we return to that below. Across the twelve completed cells
accuracy spans 49.8–58.6 % and VGR spans **−5.7 to +0.8**.

**Two of the three systems in this table cannot carry the axis at all, and the
one that can gives a negative result.** M3D-LaMed is the system §7.3 declares
uninterpretable. Qwen2.5-VL-7B is *exactly* +0.0 in all four organs,
arithmetically forced by its 100 % pair-violation rate — a model answering both
members of every pair alike cannot respond to any image manipulation, so its
+0.0 is not evidence of anything. That leaves InternVL3-8B, which passes 140/140
controls and has the lowest pair-violation rate of the montage models, and it
scores **higher with the evidence region masked in all four organs** under an
attested fill (−6.6, −8.3, −5.7, −5.7).

**The masked-region fill is a live confound, and the run that was supposed to
settle it cannot be attested.** The masked arm originally replaced the region
with the volume's 1st percentile, which on CT is air outside the patient
(≈ −1000 HU), so masking carved a cavity a model might detect as an artefact
rather than as missing evidence — a defect in our ROI definition, not a property
of the model. The intended control replaces the region with the median of a
6-voxel tissue shell instead, leaving the volume locally plausible.

That control had to be run twice. The first time, the runner took the fill from
a default that changed from `air` to `local` partway through the campaign and
wrote nothing about it into the output, so which fill produced which number was
not recoverable afterwards — a file's timestamp records when it finished, not
which default it imported. Re-run with the fill given on the command line,
recorded per row and carried in the filename:

| InternVL3-8B | fill | roi_only | roi_masked | VGR |
|---|---|---|---|---|
| Lung (n = 394) | air, 1st percentile | 53.3 % | 58.6 % | −5.3 |
| Lung (n = 394) | **local tissue median** | 53.8 % | **60.4 %** | **−6.6** |
| Colon (n = 600) | air, 1st percentile | 52.7 % | 56.8 % | −4.2 |
| Colon (n = 600) | **local tissue median** | 52.5 % | **60.8 %** | **−8.3** |
| Pancreas (n = 600) | air, 1st percentile | 53.3 % | 54.2 % | −0.8 |
| Pancreas (n = 600) | **local tissue median** | 52.7 % | **58.3 %** | **−5.7** |
| Liver (n = 600) | air, 1st percentile | 52.8 % | 58.0 % | −5.2 |
| Liver (n = 600) | **local tissue median** | 52.5 % | **58.2 %** | **−5.7** |

**Removing the artefact never shrinks the effect** (−5.3 → −6.6, −4.2 → −8.3,
−0.8 → −5.7, −5.2 → −5.7 across four organs), which is the strongest form in
which an artefact explanation can be ruled out: the cavity was working against the
result, not producing it. Pancreas shows why this had to be measured rather than
argued. Under `air` it is the one organ where InternVL3's VGR is near zero, and
we had written a paragraph explaining that exception in terms of redundant
information; under `local` it is −5.7, like the others, and the exception does
not exist.

### A.7 The capability-axis amplification, in full

**The intervention.** Amplification was to separate "signal present but
out-voted" from "no usable signal". On the full corpus it appeared to work — up
to +7.3 pp, $p<0.001$. Splitting by the confound localises the entire effect:

| Model | Growth-matched (n=1,368) | Complement (n=894) | Full corpus |
|---|---|---|---|
| Qwen2.5-VL-3B | −2.4 | **+7.3** | +2.8 |
| Qwen2.5-VL-7B | −1.2 | **+16.7** | +4.6 |
| M3D-LaMed-Phi3-4B | −1.2 | **+7.0** | +1.9 |
| Med3DVLM-7B | −0.1 | +0.0 | +0.0 |

Both subsets are exactly label-balanced (684/684 and 447/447).

**Re-run with 3.9× the data.** Because "no effect" and "not enough data to see
one" are not the same claim, we repeated the test on the per-organ runs
(8,906 items per model) after growth-matching:

| Organ | Model | n | $\alpha{=}1$ | Δ | 95 % CI | P(Δ≤0) | post |
|---|---|---|---|---|---|---|---|
| Lung | M3D | 256 | 50.8 | +2.0 | [−6.3, +8.0] | 0.567 | 52.8 |
| Lung | Qwen | 256 | 50.0 | −5.9 | [−7.6, +6.5] | 0.793 | 44.1 |
| Colon | M3D | 438 | 54.1 | +0.0 | [−7.1, +2.3] | 0.943 | 54.1 |
| Colon | Qwen | 438 | 50.0 | +0.9 | [−2.7, +8.5] | 0.197 | 50.9 |
| **Pancreas** | **M3D** | **3,074** | **45.5** | **+4.6** | **[+2.5, +6.8]** | **0.000** | **50.1** |
| Pancreas | Qwen | 3,074 | 50.0 | +0.2 | [−1.6, +1.5] | 0.517 | 50.2 |
| Liver | M3D | 1,580 | 52.4 | −0.9 | [−3.4, +1.5] | 0.920 | 51.5 |
| Liver | Qwen | 1,580 | 50.0 | +0.0 | [−3.1, +2.0] | 0.900 | 50.0 |

One cell is significant, and reading it carefully is the point: pancreas M3D moves
**45.5 % → 50.1 %**, from 4.5 pp *below* chance back *to* chance. That is a bias
correction, not recovered information — **no cell is lifted above chance.**

---

## Appendix B. The growth cue: derivation and confound audit

We found a language-accessible cue in our own corpus and report it in full,
because it bears directly on assumption **A2**.

**The cue.** Matched pairs straddle the true gap, so the larger growth amount is
necessarily the "yes" member — within a pair, the design working as intended.
Across the corpus the *marginal* distributions differ (median 26.0 mm for yes
against 15.3 mm for no) and a single threshold on the growth number, with no
image and no anatomy, reaches **69.2 %**. "Chance is exactly 50 %" bounds the
label distribution; it does **not** bound what a text-only model can score.

**The correction.** A **growth-matched subset** bins growth at 2 mm and keeps
equal yes/no per bin (n = 1,368; 684/684), on which the same rule falls to
50.8 %. Because the correction selects over already-scored items, every model is
re-reported on identical items with no additional inference.

**Auditing the rest of the question text.** Each surface feature is fitted as a
category → majority-label rule directly on the evaluation set and compared
against a permutation null (200 shuffles); only exceeding the null's 95th
percentile counts as leakage.

| Feature | Full corpus | Null p95 | Growth-matched | Null p95 |
|---|---|---|---|---|
| Target structure name | 50.0 % | 54.1 % | 54.8 % | 56.7 % |
| Organ | 50.0 % | 51.5 % | 52.6 % | 53.4 % |
| Question character length | **57.7 %** | 54.7 % | 54.0 % | 56.0 % |
| Growth amount | **69.8 %** | 56.5 % | 55.0 % | 57.1 % |
| Target × growth bin | **73.7 %** | 63.9 % | 64.7 % | 64.5 % |

Target and organ do **not** leak — observed accuracy is *below* the null, so
per-structure balancing worked and the corpus did not fail generally. Question
length *does*, at 57.7 %: the same cue casting a second shadow, since "6.2 mm"
and "13.9 mm" differ in length — a route we had not anticipated and would not
have found by auditing intentions rather than features. On the matched subset
every feature falls inside its null except target × growth, which clears its 95th
percentile by 0.2 pp and we read as noise. Without the permutation null the
matched subset's "target 54.8 %, growth 55.0 %" would read as continued leakage;
both are below what shuffled labels give at the same category count.

**Source-level regeneration does not dominate the post-hoc fix.** Inverting the
sampling — drawing growth first from a global grid, then finding geometry whose
gap straddles it — removes the growth cue exactly (threshold ceiling 50.0 %,
per-value balance exact, n = 3,860). But it requires *singleton* probes: a pair
emitting both members couples two grid values and per-value balancing cascades
through the corpus. Singletons lose the guarantee that pair members share target,
organ and volume, and the audit confirms the trade: target name then leaks at
56.0 % against a 53.9 % null. **The paired construction with post-hoc growth
matching is the protocol we recommend.**

**Scale buys sensitivity to the cue, not grounding.**

| Model | Full corpus | Growth-matched | Pair violation |
|---|---|---|---|
| Qwen2.5-VL-3B | 50.0 % | 50.0 % | 99.4 % |
| Qwen2.5-VL-7B | 50.0 % | 50.0 % | 100.0 % |
| **Qwen2.5-VL-32B** | **56.0 %** | **50.6 %** | 86.7 % |
| **InternVL3-8B** | **58.4 %** | **50.1 %** | 79.4 % |

Full-corpus accuracy tracks pair-violation rate almost exactly. Scaling 3B → 32B
within one family lowers violation 99.4 % → 86.7 % and raises full-corpus
accuracy 50.0 % → 56.0 %, while confound-free accuracy does not move. **The two
models that appear competent are exactly the two that exploited the cue.**
Reporting only the full-corpus column would have described a scaling trend toward
volumetric spatial reasoning, with three points and a monotone curve.

> **Figure 6 (new).** Growth-amount histograms by label, before and after
> matching, with threshold-rule accuracy annotated (69.2 % → 50.8 %). Two panels,
> same palette as Figure 4.

> **Figure 7 (new).** Forest plot of confound-free accuracy, one row per model,
> ordered montage-then-native as in Table §7.4, with a vertical rule at 50 %.
> Every interval crosses it; that is the figure's whole content, so the rule
> should be the strongest element on the page. Annotate each row with its image
> contribution (right margin, signed). Data: `figdata/fig7_forest_confound_free.csv`
> — columns `acc`, `ci_lo`, `ci_hi`, `image_gain`, plus `modal_share` and
> `modal_token` if the constant-response models are to be marked (Qwen-7B "no",
> LLaVA-OneVision "yes").
>
> **Figure 8 (new).** Response-channel controls: perfect questions (0–7) on the
> x-axis against "yes" rate on the y-axis, one point per model, montage and
> native in different markers. The montage cluster sits at (7, 42.9) with
> SmolVLM2 at (6, 56.4); the three native systems scatter to (3, 95.7), (2, 82.9)
> and (4, 0.0). A horizontal band at the 42.9 % the control set implies makes the
> directional bias legible. Data: `figdata/fig8_response_controls.csv`.
>
> **Figure 9 (new, carries §7.5).** The ROI four-arm decomposition: grouped bars
> per organ in the order full / roi_masked / roi_only / zero — *not* the arm
> order of the VGR definition, because the point is that full and roi_masked sit
> together while roi_only sits with zero. Bracket the two differences that matter
> (full − roi_masked, full − roi_only) with their values. Data:
> `figdata/fig9_roi_four_arm.csv`, one block per organ per model; the script
> omits any organ whose four arms are not yet equal in length rather than
> plotting an intersection.
>
> **Figure 10 (new).** Target-contrast pairs: differentiation rate (bar, left
> axis) against direction accuracy with its CI (point + whisker, right axis),
> sighted and blind side by side. The right axis must show 50 % as a rule — the
> figure exists to show the bars moving while the points do not leave the rule.
> Data: `figdata/fig10_target_pairs.csv`.

*All four CSVs are regenerated by `make_figure_data.py`, which recomputes from
the result files rather than transcribing from the tables, so a figure cannot
drift from the text beside it.*


