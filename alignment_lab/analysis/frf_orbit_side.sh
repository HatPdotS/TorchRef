#!/bin/bash
#SBATCH --job-name=orbside
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for P in 2DQ6 3K7M 1DAW 3GR5; do
  "$PY" -u alignment_lab/diagnostics/frf_orbit_side.py --pdb $P 2>&1 | grep -v "Warning\|warnings.warn" | grep -A14 "^#\|^ROW\|Traceback"
done
echo DONE
