# Experiment side — corpus generation, rendering, inference

The code that builds probes, renders inputs and scores models. Orchestration is
in [`../runs/`](../runs); analysis is in [`../analysis/`](../analysis) and
[`../paper/`](../paper).

## Rendering and its invariants

`render.py` owns the input pipeline. Two functions exist because two callers
must agree: `to_display` fixes panel orientation (sagittal unflipped, coronal
and axial mirrored horizontally) and `montage_rgb` fixes compositing (short
panels scaled up, 128-grey gutter). When the identification control and the
reader-study export each kept their own copies they diverged — the annotated
arms came out mirrored left-to-right against the published condition, a patient
left/right swap sitting inside the comparison the control exists to make.

```bash
python run_identification_control.py selftest
```

asserts on a synthetic anisotropic volume that `plain` is bit-exact against
`montage(orthogonal_views(...))`, that all four conditions emit identically
shaped input, that `identified` is byte-identical to `render_case`, and that the
annotated arms agree with `plain` outside their annotation. **Run it after any
change here.** The obvious alternative check — does the control arm reproduce the
published accuracy — cannot fail, because both sides are at chance whether the
renderer is right or wrong.

## Runners

| file | experiment |
|---|---|
| `run_identification_control.py` | E1 four input conditions, and E2's ceiling arms (`geometry-oracle`, `numeric-oracle`, `text-oracle`, and the decoding/framing controls) |
| `run_subtasks.py` | E3 perceive / name / measure, with a no-annotation control |
| `run_input_richness.py` | E9 how much volume is shown, annotation held fixed |
| `run_inference_compute.py` | E5 chain-of-thought, self-consistency, self-verification |
| `run_leakage.py` | E12 pretraining overlap, with a positive control |
| `run_multimodel.py` | the published multi-model montage evaluation; `MODEL_ID` is the model registry |
| `run_roi_arms.py` | four-arm grounding; records the mask fill per item |
| `sanity_controls.py` | known-answer response-channel controls, `--framing native\|montage` |
| `export_reader_study.py` | reader-study export; shares the render rules above |
| `make_seg_cache.py` | regenerates `cfqa_*/seg_cache` (lock-coordinated, resumable) |
| `verify_seg_provenance.py` | checks a regenerated cache against the stored `gap_mm` |

## Scoring

Every arm scores by likelihood over the option strings, never by parsing free
text — 3D medical VLMs answer in referring-expression templates whose wording
embeds the answer words. `MontageModel.score` reads only the option's own logit
rows; casting the whole sequence × 152k-vocabulary tensor to float32 to read a
few rows was 1.8 GB per option per probe and the allocation that OOM'd
co-scheduled jobs.

The `-gen` variants decode greedily and parse instead, and exist because a
likelihood rule can pin argmax to one option for every item — which yields
exactly 50% on a balanced corpus and is indistinguishable from "at chance" in any
accuracy table. Reported side by side, the pair says which one you are looking at.
