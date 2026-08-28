#!/bin/bash
# Layer A of the E-value work: the property report, paired with the functional
# panel that decides. Run over structures spanning primitive and centred
# lattices and low to high symmetry, because the epsilon and Wilson-shape
# properties are properties of real reflection sets.
#SBATCH --job-name=etable
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-4
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
# 1DAW is C2 (centred, so eps != 1 everywhere), 2DQ6 is the tNCS case whose
# moment ratio reads ~5.5, 3K7M and 3A5V are the high-symmetry ends, 6G9X is
# where the rescore reproducibly fails.
PDBS=(1DAW 2DQ6 3K7M 3A5V 6G9X)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "### $PDB on $(hostname)"
"$PY" -u alignment_lab/analysis/e_table.py --pdb "$PDB" 2>/dev/null
echo "RC=$?"
