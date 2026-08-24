#!/bin/bash
# Stage D acceptance panel: 10 benchmark structures x 10 seeded trials, run in
# BOTH worktrees so the comparison is paired per seed. One array task per
# structure; OLD and NEW run back to back inside the task on the same node.
#SBATCH --job-name=d_panel
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-9
set -uo pipefail
NEW=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
OLD=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/_stagea_baseline
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
export TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) pdb=$PDB"
for tree in OLD NEW; do
  eval "root=\$$tree"
  cd "$root"
  PYTHONPATH="$root" "$PY" -u "$NEW/alignment_lab/analysis/panel_ranks.py" \
    --pdb "$PDB" --trials 10 --tag "$tree" 2>/dev/null | grep '^ROW '
  echo "tree=$tree rc=$?"
done
