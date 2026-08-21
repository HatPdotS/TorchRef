#!/bin/bash
# Part 1 of the FRF cleanup: settle lmax_cap / anisotropy / orbit-unroll /
# Patterson-radius by measurement before the switches are deleted.
#
# One array task = one (structure, trial) cell, running every arm in the same
# process so the paired comparison against `production` is exact.
#
#   sbatch --array=0-99 --partition=hour --time=00:55:00 --cpus-per-task=4 \
#          --mem=32G alignment_lab/analysis/config_sweep_array.sh 1
#
# 10 structures x 10 trials = 100 tasks. Pass the stage (1 or 2) as $1; any
# further arguments go through to the diagnostic.
#SBATCH --job-name=frf_cfg_sweep
#SBATCH --output=alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=alignment_lab/slurm/%x_%A_%a.err
set -uo pipefail

STAGE="${1:?usage: config_sweep_array.sh <stage> [extra args...]}"
shift || true

REPO="${FRF_SWEEP_REPO:-/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement}"
PY="$REPO/.dev/bin/python"
[ -x "$PY" ] || PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python

cd "$REPO"
export PYTHONPATH="$REPO"
export TORCHREF_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$TORCHREF_NUM_THREADS"
export MKL_NUM_THREADS="$TORCHREF_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

# Worklist: keep in step with lab.benchmark.BENCH_PDBS (order is a seed contract).
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
TRIALS=10
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PDB="${PDBS[$((IDX / TRIALS))]}"
TRIAL=$((IDX % TRIALS))

OUTDIR="alignment_lab/runs/config_sweep_s${STAGE}_${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUTDIR" alignment_lab/slurm

echo "task $IDX -> $PDB trial $TRIAL stage $STAGE -> $OUTDIR"
echo "repo=$REPO  sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
rc=0
"$PY" -u alignment_lab/diagnostics/frf_config_sweep.py \
    --pdb "$PDB" --trial "$TRIAL" --stage "$STAGE" \
    --out-csv "$OUTDIR/${PDB}_t${TRIAL}.csv" "$@" || rc=$?

# Report the real exit status: `rc=$?` must follow the command directly, or a
# task that dies gets logged COMPLETED.
echo "exit_code=$rc"
exit "$rc"
