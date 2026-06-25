# Extended Figure 4 — single-core runtime (Figure 2c at 1 CPU core)

Reproduces **Figure 2c** (the wall-clock runtime box plot) with every program pinned to a
**single CPU core**, over the **full conserved AlphaFold-start benchmark (n = 715)**.

**Why 1 core.** The main Figure-2 benchmark ran all three programs at **4 cores** each
(`run_af_pipeline.py` `--cpus-per-task=4` / `TORCHREF_NUM_THREADS=4`; `phenix_refine.sh`
`refinement.main.nproc=4`; REFMAC has no parallelism). This panel re-levels them onto a
clean per-core, single-threaded comparison — it mainly affects TorchRef vs PHENIX; REFMAC
is essentially unchanged because it doesn't thread.

Each program refines the same Phaser-placed AlphaFold model (reused from
`../../figure2_alphafold_start/placed/`) for 10 macro cycles, pinned to 1 thread, at
`--mem 8G` (matching the main benchmark). The conserved set is frozen in
`codes_conserved.txt` (copied from `../../figure2_alphafold_start/runs/metrics/conserved_codes.txt`),
so the box plots are the **same sample** as Fig 2c, only re-timed at 1 core.

## Workflow

```bash
PY=../../../.dev/bin/python   # repo-local editable install

# 1. inspect one example sbatch per program
$PY submit_singlecore.py --dry-run --limit 1

# 2. submit all 2145 jobs (715 structures x 3 programs), all fresh
$PY submit_singlecore.py

# 3. once jobs finish, collect timings -> data/exF4_singlecore.csv
$PY aggregate_singlecore.py

# 4. plot -> output/exF4_singlecore_runtime.png
$PY plot_singlecore.py
```

`submit_singlecore.py` flags: `--programs`, `--codes`, `--limit`, `--dry-run`. All cells
are submitted fresh (no resume-skip — an existing `timing.txt` is overwritten) so the
whole set is timed under identical conditions.

## How the single core is enforced

Every job requests `#SBATCH --cpus-per-task=1` and pins the program to 1 thread:
- **TorchRef** — `export TORCHREF_NUM_THREADS=1` (`--device cpu`)
- **PHENIX** — `refinement.main.nproc=1`
- **REFMAC** — `export OMP_NUM_THREADS=1`

Partitions/walltime: TorchRef/PHENIX on `day` (24 h), REFMAC on `hour` (1 h) — generous
so timing is never truncated. `aggregate_singlecore.py` reports any `rc!=0` / not-yet-run
cell so a straggler (e.g. an unusually large structure) can be resubmitted to `week`.

## Timing

Each job times the program with a uniform shell timer (`date +%s.%N` before/after the
call) and writes `wall_s <float>` + `rc <int>` to `runs/{program}/{code}/n1/timing.txt`.
This single wall-clock metric is identical across all three engines (no per-program log
parsing). `aggregate_singlecore.py` drops cells with a nonzero `rc` and keeps only the
**conserved-at-1-core** set (codes all three programs solved at 1 core); the kept codes
go to `data/singlecore_conserved.txt`.

## Panel

- **exF4** — per-engine wall-clock distribution at 1 core (box plot, log y, minutes;
  5th–95th-percentile whiskers, no outliers, faint jittered strip of individual
  structures), engines ordered fastest → slowest (Refmac, TorchRef, PHENIX). Colors match
  the paper (Refmac `#762a83`, TorchRef `#b2182b`, PHENIX `#2166ac`). Annotation reports
  `n`, TorchRef-vs-Refmac and TorchRef-vs-PHENIX paired median ratios.

## Files

```
codes_conserved.txt        # frozen n=715 conserved codes (= Fig 2c set)
submit_singlecore.py       # sbatch 2145 single-core jobs (3 programs x 715)
aggregate_singlecore.py    # parse timing.txt -> data/exF4_singlecore.csv
plot_singlecore.py         # box plot -> output/exF4_singlecore_runtime.png
data/exF4_singlecore.csv   # program,code,wall_s
data/singlecore_conserved.txt
runs/{program}/{code}/n1/  # out.log + timing.txt per cell (refmac also refmac.log)
output/exF4_singlecore_runtime.png
_scaling_archive/          # the previous CPU-core-scaling experiment (superseded)
```

## Caveats

- SLURM may place cells on heterogeneous CPU types, so absolute times carry node-to-node
  variance; the robust signal is the per-engine *distribution* and the *paired* median
  ratios (each ratio is computed per-structure, so node variance largely cancels).
- The earlier CPU-core-scaling experiment (1–16 cores, 10 structures, + GPU and overhead
  panels) lives in `_scaling_archive/` for reference.
