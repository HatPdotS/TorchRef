#!/bin/bash
# Every CAND line of the pose panel's trial 0 on all ten structures, written to
# alignment_lab/runs/cand_<label>.txt so a refactor can be checked bitwise
# against them. Usage: sbatch cand_baseline.sh <label>
#SBATCH --job-name=candbase
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-9
set -uo pipefail
LABEL=${1:-baseline}
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
PDBS=(1DAW 3E98 3A5V 3VRJ 1AK5 3K7M 3GR5 2DQ6 4BX9 6G9X)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/runs
"$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial 0 --arms llg --verbose 2 2>/dev/null \
  | grep -E "^CAND|^ROW" | sed "s/^/$PDB /" > "alignment_lab/runs/cand_${LABEL}_${PDB}.txt"
wc -l "alignment_lab/runs/cand_${LABEL}_${PDB}.txt"
