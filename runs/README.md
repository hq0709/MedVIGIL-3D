# Experiment side — orchestration

Everything that *runs* jobs. No analysis lives here; the paper side is
[`../paper/`](../paper). The experiment code these scripts launch is in
[`../spatialgen/`](../spatialgen).

| file | what it does |
|---|---|
| `run_queue.py` | VRAM-aware job queue. Atomic per-job claims, so one queue per card can share a single job list. `--share-headroom` computes the budget from what *other users* hold plus a promised margin, instead of from free memory. `--pause-file` lets a run that needs both cards take them without killing the queue. |
| `make_jobs.py` | job lists for the identification control and the ceiling arms |
| `make_program.py` | job lists for input richness, leakage, native-model framing, Aria, grounding |
| `prerender.py` | fills the render cache on CPU, in parallel |
| `audit_conditions.py` | what each condition actually shows, measured on pixels with no model |
| `summarise.py` | every result table in one command |
| `director.sh` | runs the waves in order, recomputing each card's budget per wave |
| `e4_watcher.sh` | waits for a card with no co-tenant, then gives the 72B both cards |
| `deadline.sh` | stops launching at a soft deadline and stops the run at a hard one |
| `jobs_e4_72b.sh` | the 72B image arms; needs both cards sharded |

Job lists (`jobs_*.jsonl`) are **regenerated, not stored** — they are scheduling
artefacts, and keeping them invites running a stale one.

## Typical use

```bash
source /path/to/env.sh                 # see ../DATA.md §5
python runs/make_program.py            # writes runs/jobs_*.jsonl
python runs/run_queue.py --jobs runs/jobs_e9.jsonl --gpus 6,7 \
       --share-headroom 50 --skip-existing
```

Jobs write to `<out>.part` and are renamed only on exit 0, so an interrupted run
loses nothing and `--skip-existing` resumes it.

## Two operational rules that cost us time to learn

**Do not schedule against free memory on a shared card.** A neighbour's training
job sitting at 10 GB may still climb to 47 GB; taking the rest of the card means
their run dies, not ours. `--share-headroom 50` encodes the promise.

**Purge the render cache after any renderer change, and re-run the parity
selftest.** A stale cache feeds the old images to every job and the accuracies
will not reveal it.
