#!/bin/bash
# End-to-end pose recovery, which is the only metric that is actually the
# deliverable. Rank is a proxy; this is not.
#
# Two questions in one array: does dropping the rescore cost anything, and is
# n_rotation_candidates=15 leaving coverage on the table? The FRF's worst truth
# rank over 100 cells is 21, so 15 carries 93% and 25 carries 100% -- but
# coverage is not recovery, and only this measures recovery.
#SBATCH --job-name=posearm
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-9
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for T in 0 1 2; do
  "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial "$T" \
    --arms m_letf1,none --n-rotation-candidates 15 2>/dev/null | grep '^ROW '
  "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial "$T" \
    --arms none --n-rotation-candidates 25 2>/dev/null | grep '^ROW '
done
