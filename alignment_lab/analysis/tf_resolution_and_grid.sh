#!/bin/bash
#SBATCH --job-name=tfgrid
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$REPO"
for P in 1DAW 3E98 3A5V 3GR5 1AK5 3K7M 2DQ6 4BX9 6G9X; do
  "$PY" -u alignment_lab/diagnostics/tf_resolution_and_grid.py --pdb "$P" --threads 4 2>&1 \
    | grep -E "^#|^ROW|Traceback|Error" || echo "# $P FAILED"
done
