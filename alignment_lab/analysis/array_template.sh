#!/bin/bash
# SLURM array template for the alignment lab.
#
# Resources go on the sbatch command line, not in this file, so one template
# serves CPU diagnostics and GPU sweeps. The array index selects a (pdb, trial)
# cell from the worklist below.
#
#   sbatch --array=0-29 --partition=hour --time=00:55:00 --cpus-per-task=4 \
#          --mem=32G alignment_lab/analysis/array_template.sh ghost_origin
#
# 10 structures x 3 trials = 30 tasks. Note the +-4-6 seed-to-seed rank spread:
# 3 trials is for a smoke run, ~10 for anything you intend to believe.
#SBATCH --job-name=align_lab
#SBATCH --output=alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=alignment_lab/slurm/%x_%A_%a.err
set -euo pipefail

DIAG="${1:?usage: array_template.sh <diagnostic-name> [extra args...]}"
shift || true

REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY="$REPO/.dev/bin/python"
[ -x "$PY" ] || PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python

cd "$REPO"
export PYTHONPATH="$REPO"
export TORCHREF_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$TORCHREF_NUM_THREADS"
export MKL_NUM_THREADS="$TORCHREF_NUM_THREADS"
export PYTHONUNBUFFERED=1
[ -z "${SLURM_JOB_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=""

# Worklist: keep in step with lab.benchmark.BENCH_PDBS (order is a seed contract).
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
TRIALS=3
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PDB="${PDBS[$((IDX / TRIALS))]}"
TRIAL=$((IDX % TRIALS))

OUTDIR="alignment_lab/runs/${DIAG}_${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUTDIR" alignment_lab/slurm

echo "task $IDX -> $PDB trial $TRIAL -> $OUTDIR"
rc=0
"$PY" -u "alignment_lab/diagnostics/${DIAG}.py" \
    --pdb "$PDB" --trial "$TRIAL" \
    --out-csv "$OUTDIR/${DIAG}_${PDB}_t${TRIAL}.csv" "$@" || rc=$?

# Report the real exit status: a task that dies must not be logged COMPLETED.
echo "exit_code=$rc"
exit "$rc"
