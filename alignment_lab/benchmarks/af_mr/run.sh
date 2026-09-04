#!/bin/bash
# The AlphaFold molecular-replacement benchmark: place each structure's
# predicted search models with the pipeline, refine the result, and refine
# Phaser's placement of the same models identically. One array task per
# structure; results land in runs/<code>/summary.json, read by table.py.
#
#   sbatch alignment_lab/benchmarks/af_mr/run.sh          # 15-3 A, the default
#   D_MIN=4.0 TAG=dmin4 sbatch alignment_lab/benchmarks/af_mr/run.sh
#SBATCH --job-name=afbench
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=day
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-49
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
BENCH=$REPO/alignment_lab/benchmarks/af_mr
D_MIN=${D_MIN:-3.0}
D_MAX=${D_MAX:-15.0}
TAG=${TAG:-dmin${D_MIN}}
N_CYCLES=${N_CYCLES:-10}

# Task i takes the i'th code, comments and blank lines skipped.
CODE=$(grep -v '^#' "$BENCH/worklist.txt" | grep -v '^[[:space:]]*$' \
       | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$CODE" ] || { echo "BENCH task=$SLURM_ARRAY_TASK_ID EMPTY"; exit 0; }

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab"
export TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
# Concurrent array tasks poison a shared numba __pycache__; give each its own.
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

OUT=$BENCH/runs/$TAG/$CODE
mkdir -p "$OUT"
"$PY" -u alignment_lab/diagnostics/af_placement.py "$CODE" \
      --d-min "$D_MIN" --d-max "$D_MAX" --n-cycles "$N_CYCLES" --out "$OUT" 2>&1 \
  | grep -v "Warning\|warnings.warn\|ModelFT copied\|Parametrization built\|SfFFT:" \
  | grep -A20 "^PLACED\|^ROW\|Traceback"
echo "BENCH code=$CODE tag=$TAG rc=$?"
