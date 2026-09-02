#!/bin/bash
#SBATCH --job-name=tps
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-2
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
case $SLURM_ARRAY_TASK_ID in
  0) "$PY" -u alignment_lab/diagnostics/truth_pose_scores.py --pdb 2DQ6 --trial 3 ;;
  1) "$PY" -u alignment_lab/diagnostics/truth_pose_scores.py --pdb 2DQ6 --trial 0 ;;
  2) "$PY" -u alignment_lab/diagnostics/truth_pose_scores.py --pdb 3GR5 --trial 0 ;;
esac
echo DONE
