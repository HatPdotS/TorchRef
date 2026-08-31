#!/bin/bash
#SBATCH --job-name=ftfdisc
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-9
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Concurrent array tasks otherwise poison a shared __pycache__ for numba.
export NUMBA_CACHE_DIR="/tmp/numba_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$REPO"
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
P=${PDBS[$SLURM_ARRAY_TASK_ID]}
"$PY" -u alignment_lab/diagnostics/frf_vs_ftf_discrimination.py \
  --pdb "$P" --trials 3 --n-cand 25 --n-rotation-peaks 200 \
  --out-csv "alignment_lab/runs/ftf_disc_${SLURM_ARRAY_JOB_ID}.csv" 2>&1
echo "RC=${PIPESTATUS[0]} pdb=$P"
