# Multi-city pipeline (round 4) — runbook

> **What this is.** A single orchestrator that runs the round-3
> X-SAGE framework over the TIST2015 multi-city dataset, with tqdm,
> human-readable logging, and resumable checkpoints. Designed to be
> launched from the VS Code terminal and walked away from.

## Quick start

```bash
cd /Users/lucaaliberti/Downloads/IntentAwareRS_thesis

# Full real run (5 cities, all stages) — overnight job, ~4–6 h
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages all

# Just one city, one stage (e.g. re-do Istanbul Stage A only)
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities istanbul --stages stageA --force

# Resume after sleep/crash — same command as a fresh run; done-markers
# auto-skip completed (city, stage) pairs
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages all
```

## Smoke test (validate plumbing, ~1–2 min)

```bash
.venv/bin/python -m experiments.multicity.run_multicity --smoke
```

Uses a 200-user subset of `nyc_tist.tsv` and runs the full chain
(step01 → backbones → stageA/B/C). Should print a status table at the
end. **Always run this once after pulling.**

## Flags

| flag | default | meaning |
|---|---|---|
| `--cities` | `all` | Comma list. Choices: `istanbul, bangkok, nyc_tist, saopaulo, tokyo_tist`. |
| `--stages` | `all` | Comma list. Choices: `carve, step01, backbones, stageA, stageB, stageC, stageD`. |
| `--force`  | off | Re-run (city, stage) even if done-marker exists. |
| `--smoke`  | off | Tiny dry-run on a 200-user subset of `nyc_tist`. |
| `--smoke-users` | 200 | User cap for smoke mode. |
| `--seed`   | 42 | Reproducibility seed for step01 / stage runners. |

## Stages

1. **`carve`** — one-pass extraction of all selected cities from the
   TIST2015 global files (`data/raw/dataset_TIST2015_{Checkins,POIs,
   Cities}.txt`) to TSMC2014-format TSVs under `data/raw_tist/<city>.tsv`.
   Window: **Apr 2012 – Feb 2013** (compromise with TSMC NYC/TKY).
2. **`step01`** — frozen `pipeline.step01_preprocessing.preprocess_tsmc2014`:
   k-core=10, 80/10/10 per-user temporal split, derived features.
   Output: `data/processed/<city>/`.
3. **`backbones`** — floor FM (`experiments.run_baselines --models FM`)
   + tuned B_full (`experiments.round2_tune_bfull`). Output:
   `outputs/<city>/baselines/` + `outputs/<city>/xsage/backbone/`.
4. **`stageA`** — situation clustering (rough k-means; the cardinal
   X-SAGE structural read). Output: `outputs/<city>/xsage/situations/`.
5. **`stageB`** — per-situation fairness lens (LT, KL, sinks).
6. **`stageC`** — projection: macro transition matrix, F1 vs clock,
   McNemar.
7. **`stageD`** — three-way comparison (B_blind / X-SAGE / B_full) +
   matched-OFF TOST.

Stages run in the listed order; later stages depend on earlier ones.
`--stages all` runs every stage. Re-running with the same flags is
safe: completed (city, stage) pairs are skipped automatically.

## Where output lands

* **Heavy per-city artefacts**: `outputs/<city>/` (same convention as
  TSMC NYC/TKY/TKY_BAL — produced by frozen modules). The round-3
  `outputs/NYC` and `outputs/TKY` are NOT touched.
* **Per-city done markers + headline summaries**:
  `outputs_multicity/<city>/{.done_<stage>, summary_<stage>.json}`.
* **Live logs**: `outputs_multicity/logs/run_<timestamp>.log`. Each
  major step prints a one-liner like `[istanbul] step01: k-core
  converged, 30142 users / 18003 venues kept` AND the headline
  numbers as they're computed.

## Resumability

If your laptop sleeps, the run crashes, or you Ctrl-C, **just re-run
the same command**. Each (city, stage) wrote a done-marker on success;
the orchestrator skips done pairs and continues from the first
incomplete one. Use `--force` to ignore the markers.

If a stage failed, the marker is `.fail_<stage>` (not `.done_<stage>`)
under `outputs_multicity/<city>/`. Inspect it, fix the cause, then
re-run with `--force` (or delete the fail marker).

## Time estimates

The orchestrator prints rough per-city ETAs at the start of every run.
Reference (5 cities, all stages):

| city | post-kcore users | estimated wallclock |
|---|---:|---:|
| istanbul   | 22 631 | ~3–4 h (largest, dominates total) |
| bangkok    | 6 316  | ~30–45 min |
| tokyo_tist | 7 160  | ~35–50 min |
| saopaulo   | 4 395  | ~25–35 min |
| nyc_tist   | 4 113  | ~20–30 min |
| **total**  |        | **~4.5–6 h** |

Most of the time goes to B_full BPR training (`backbones`); Stage A/C
are cheap. **NOTE on dependencies**: Stage B (fairness lens) and Stage
D (three-way) both need the floor FM scores from `backbones` — they
read `outputs/<city>/baselines/FM.best_hp.json`. Stage A and Stage C
do NOT need backbones (only step01).

Fast structural-only pass (no backbones training, ~15 min on all 5
cities — completes Stage A and Stage C; skips Stage B):

```bash
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages step01,stageA,stageC
```

To add Stage B / D afterwards (requires backbones — adds ~1–2 h per
city on the bigger ones):

```bash
.venv/bin/python -m experiments.multicity.run_multicity \
    --cities all --stages backbones,stageB
```

The orchestrator now warns up front if you request Stage B/D without
backbones AND the floor-FM artefact isn't on disk, so you don't waste
a run.

## Round-3 isolation

Frozen round-3 artefacts stay untouched:

* `outputs/NYC/`, `outputs/TKY/`, `outputs/TKY_BAL/`
* `pipeline/`, `engine/`
* All round-3 reports, the `round3-complete` tag

This orchestrator only writes to:

* `data/raw_tist/`, `data/processed/<new_city>/`, `outputs/<new_city>/`,
  `outputs_multicity/`

## Troubleshooting

* **"X-SAGE Stage A failed rc=1"**: usually means step01 hasn't run
  for that city. Run `--stages step01 --cities <city>` first.
* **Out of memory on Istanbul**: it's the biggest city (22k users,
  1.26M interactions). Close other apps. The b_full BPR is ~2GB at
  d=128.
* **`.fail_step01` marker stuck**: probably the carved TSV is empty
  or truncated. Delete the marker, the carved TSV, and the
  `.done_carve` marker; re-run.

## Round-4 pre-registered predictions

Before running any stages, see `outputs_multicity/PREDICTIONS_MULTICITY.md`
— the R8 falsifiable predictions for each city. Do NOT edit them
after seeing results.
