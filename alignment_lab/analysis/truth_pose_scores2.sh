#!/bin/bash
#SBATCH --job-name=tps2
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-4
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
T="$PY -u alignment_lab/diagnostics/truth_pose_scores.py"
case $SLURM_ARRAY_TASK_ID in
  0) $T --pdb 2DQ6 --trial 3 --tf-d-min 4.0 --tf-d-max 15.0 ;;
  1) $T --pdb 2DQ6 --trial 3 --tf-d-min 3.0 ;;
  2) $T --pdb 4BX9 --trial 0 ;;
  3) $T --pdb 6G9X --trial 0 ;;
  4) $T --pdb 1DAW --trial 0 ;;
esac
echo DONE
