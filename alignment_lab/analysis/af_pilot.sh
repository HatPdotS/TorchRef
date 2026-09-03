#!/bin/bash
# AlphaFold-start molecular replacement pilot: place each structure's processed
# AlphaFold search model(s) with TorchRef's pipeline, refine that placement and
# Phaser's placement with the same recipe in the same job.
#SBATCH --job-name=afmr
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-23
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
WORKLIST=${WORKLIST:-$REPO/alignment_lab/analysis/af_pilot_worklist.txt}
read CODE NCOMP NCOP <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WORKLIST")"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
echo "CELL code=$CODE components=$NCOMP copies=$NCOP"
"$PY" -u alignment_lab/diagnostics/af_placement.py "$CODE" ${AF_EXTRA:-} 2>&1 \
  | grep -v "Warning\|warnings.warn\|ModelFT copied\|Parametrization built\|SfFFT:" | grep -A15 "^PLACED\|^ROW\|Traceback"
