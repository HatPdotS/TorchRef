#!/bin/bash
# Placement only, at a finer high-resolution limit: does the window explain the
# two hexagonal misses without costing the rest?
#SBATCH --job-name=afwin
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-23
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
read CODE NCOMP NCOP <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$REPO/alignment_lab/analysis/af_pilot_worklist.txt")"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
for DM in 3.0 3.5; do
  "$PY" -u alignment_lab/diagnostics/af_placement.py "$CODE" --d-min $DM --no-refine \
    --out $REPO/alignment_lab/runs/af_mr_dmin${DM}/$CODE 2>&1 \
    | grep -v "Warning\|warnings.warn\|ModelFT copied\|Parametrization built\|SfFFT:" | grep -A15 "^ROW\|Traceback"
done
