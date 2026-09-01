#!/bin/bash
# Is 2DQ6 a cell the pipeline solves, or a coin flip the panel samples once?
#
# 2DQ6 t1 moved 6.5 -> 28.7 deg on a 1.3e-5 change in Sigma(s), taking the panel
# from 30/30 to 29/30. If the structure is genuinely marginal then its pass rate
# over SEEDS is intermediate, and a single flip carries no information about the
# change that produced it -- a different seed would have flipped it anyway.
# Three trials per structure cannot tell the difference; ten can.
#SBATCH --job-name=marg
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-3
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
# Two known-marginal structures and two the panel has never lost, as controls.
PDBS=(2DQ6 6G9X 1DAW 3K7M)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for T in $(seq 0 9); do
  "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial "$T" \
    --arms llg,analytic_r,corr --n-rotation-candidates 25 2>/dev/null | grep '^ROW '
done
