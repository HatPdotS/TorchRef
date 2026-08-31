#!/bin/bash
# End-to-end pose recovery, which is the only metric that is actually the
# deliverable. Rank is a proxy; this is not.
#
# 10 structures x 3 trials x 2 arms. The arms are the open ranking question:
# analytic R (the default) against the translation LLG, which wins 27/30 to
# 22/30 at rank level.
#
# 25 candidates and no early stopping, matching the pipeline. Under the old rule
# it walked the list until a placement beat R < 0.45 and returned that, so the
# answer depended on FRF order and could not be compared against a harness that
# ranks the whole list -- 2DQ6 solved 6/10 end to end while truth was top-ranked
# by analytic R in 0/10, which is only possible if the two measure different
# things.
#SBATCH --job-name=posearm
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=day
#SBATCH --time=04:00:00
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
    --arms analytic_r,llg_tf --n-rotation-candidates 25 2>/dev/null | grep '^ROW '
done
