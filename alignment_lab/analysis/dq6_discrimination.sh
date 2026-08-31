#!/bin/bash
# On the seeds where 2DQ6 fails, is truth in the candidate list at all?
#
# The structure passes 6/10 end to end and bimodally -- 0-4 deg or 21-31. That
# is either the rotation function never producing the true orientation, or the
# translation function producing it and ranking the tNCS alternative above it.
# Those are different problems and the panel cannot tell them apart, because it
# only reports the winner. This reports truth's RANK under each score.
#SBATCH --job-name=dq6disc
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
# 15 candidates: what the pipeline actually carries into the translation search.
"$PY" -u alignment_lab/diagnostics/frf_vs_ftf_discrimination.py \
  --pdb 2DQ6 --trials 10 --n-cand 15 2>/dev/null | grep -E '^(ROW|CAND)'
echo DONE
