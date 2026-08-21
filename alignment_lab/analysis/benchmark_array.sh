#!/bin/bash
# The rotation search's standing benchmark: accuracy, memory and runtime.
#
# One array task per structure, all arms and trials in one process on an
# EXCLUSIVE node, so the arm comparison is within-node and the memory peaks are
# not another job's. Runtime on a shared node measures the node; every row also
# carries a calibration workload and the host identity so that is checkable
# rather than assumed.
#
#   sbatch --array=0-9 --partition=hour --time=00:55:00 --exclusive \
#          --mem=200G alignment_lab/analysis/benchmark_array.sh \
#          --arms cap48,cap64,cap100 --trials 3
#
# --mem must cover the largest arm: cap100 on the P432 structures needs well
# over 32 GB, which is what the OOMs in job 489988 were.
#SBATCH --job-name=frf_bench
#SBATCH --output=alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=alignment_lab/slurm/%x_%A_%a.err
set -uo pipefail

REPO="${FRF_BENCH_REPO:-/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement}"
PY="$REPO/.dev/bin/python"
[ -x "$PY" ] || PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python

cd "$REPO"
export PYTHONPATH="$REPO"
export TORCHREF_NUM_THREADS="${FRF_BENCH_THREADS:-4}"
export OMP_NUM_THREADS="$TORCHREF_NUM_THREADS"
export MKL_NUM_THREADS="$TORCHREF_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

# Pin the thread count rather than inheriting it from the allocation: an
# exclusive node hands over every core, so SLURM_CPUS_PER_TASK would make the
# timings depend on the node's size instead of on the code.
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PDB="${PDBS[$IDX]}"

OUTDIR="alignment_lab/runs/frf_bench_${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUTDIR" alignment_lab/slurm

echo "task $IDX -> $PDB on $(hostname), ${TORCHREF_NUM_THREADS} threads"
echo "repo=$REPO sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
rc=0
"$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
    --pdb "$PDB" --out-csv "$OUTDIR/${PDB}.csv" "$@" || rc=$?

# `rc=$?` has to follow the command directly, or a task that dies reads COMPLETED.
echo "exit_code=$rc"
exit "$rc"
