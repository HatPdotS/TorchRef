#!/bin/bash
# The 10 x 3 panel with the pose gate (rotation AND translation), at the
# pipeline's default translation window and with the window removed. The
# default is now the rotation search's own window; "full" is what it used to be.
#SBATCH --job-name=ptrans
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-9
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for T in 0 1 2; do
  "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial $T --arms llg \
    2>/dev/null | grep '^ROW ' | sed 's/^ROW/ROW window=default/'
  "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial $T --arms llg \
    --tf-d-min 0 --tf-d-max inf 2>/dev/null | grep '^ROW ' | sed 's/^ROW/ROW window=full/'
done
echo DONE
