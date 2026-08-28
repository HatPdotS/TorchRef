#!/bin/bash
# Re-establish the FRF panel after the merge, and settle the hydrogen question in
# the same pass. Every previously published FRF number was measured on
# hydrogen-free structure factors; dev now keeps hydrogens by default, so the old
# 98/100 is not a baseline any more.
#SBATCH --job-name=rebase
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
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
echo "host=$(hostname) pdb=$PDB"
"$PY" -u alignment_lab/analysis/panel_ranks.py --pdb "$PDB" --trials 10 --tag withH 2>/dev/null | grep '^ROW '
"$PY" -u alignment_lab/analysis/panel_ranks.py --pdb "$PDB" --trials 10 --tag noH --exclude-h 2>/dev/null | grep '^ROW '
