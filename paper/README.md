# Paper side — the manuscript and the code that produces its numbers

Everything here either *is* the manuscript or recomputes a number that appears
in it. Nothing here launches a model. If you want to reproduce a figure or check
a claim, this is the only directory you need; the experiment harness is under
[`../runs/`](../runs) and [`../spatialgen/`](../spatialgen).

| file | what it is |
|---|---|
| `SECTION_3D.md` | **authoritative volumetric section.** Every number recomputed from `../results_new/`. |
| `PAPER.md` | full internal draft. Chapter 7 predates the identification control and the sub-task decomposition; it says so at its head and points here. Not the submission. |
| `examples/` | example renderings of the input-richness arms, for figure material |
| `ref2d/` | frozen 2D reference numbers the cross-modality section reads |

## Reproducing the numbers

These run on the committed data alone — **no MSD volumes, no model weights, no
GPU**:

```bash
cd ..                                        # scripts resolve paths from the repo root
python growth_matched.py                     # growth-matched subset + the 13-model table
python verify_paper_tables.py                # asserts the headline and ROI tables in PAPER.md
python analysis/verify_claims.py             # modal share, AUROC, confound ceilings, margin gate
python analysis/identification_control_ci.py # E1/E2/E3 with volume-clustered CIs
python runs/summarise.py                     # every table in this section, in one command
python make_figure_data.py                   # regenerates figdata/ from the raw predictions
python check_integrity.py                    # arm-length and pairing integrity across all runs
```

`runs/summarise.py` writes what `results_new/FINAL_SUMMARY.txt` holds;
`analysis/identification_control_ci.py` writes `FINAL_CI.txt`;
`analysis/verify_claims.py` writes `FINAL_CLAIMS.txt`. All three are committed,
so a reader can check a number without running anything.

## Which claim comes from which file

| section of `SECTION_3D.md` | recomputed by |
|---|---|
| §1 corpus verification (100% on 8,476) | `runs/summarise.py` §2, `results_new/id_*_geometry_geometry-oracle.jsonl` |
| §2 identification control, scale | `analysis/identification_control_ci.py` |
| §2 input audit (42.0% lesion absent) | `runs/audit_conditions.py` → `results_new/condition_audit.jsonl` |
| §3 sub-task decomposition | `runs/summarise.py` §4 |
| §4 ceiling and the text collapse | `runs/summarise.py` §2 |
| §5 input richness | `runs/summarise.py`, `results_new/id_*_richness-*.jsonl` |
| §6 modal share, gap/growth ceilings, margin gate | `analysis/verify_claims.py` |
| §7 render parity | `python spatialgen/run_identification_control.py selftest` |

## Two things a reviewer will ask, and where the answer is

**"Is your control arm really the published condition?"** — `selftest` asserts
that `plain` is bit-exact against `montage(orthogonal_views(...))`, that all four
conditions emit identically shaped input, and that `identified` is byte-identical
to the reader-study rendering. Accuracy comparisons cannot establish this: two
different renderings both score 50%.

**"Are you sure you are measuring vision?"** — §4. The same collapse reproduces
with the distance supplied in text and the image removed entirely.
