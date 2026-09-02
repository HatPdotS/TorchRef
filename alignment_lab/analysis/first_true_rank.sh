#!/bin/bash
# How deep in the rotation function's (de-duplicated) list does the true
# orientation sit? The SOLN table carries each solution's FRF index k and its
# truth flag; the smallest k flagged true per cell is the shortlist depth the
# translation search needed.
#SBATCH --job-name=ftr
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
for T in 0 1 2 3 4; do
  "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb "$PDB" --trial $T --arms llg --verbose 2 2>/dev/null \
    | awk -v pdb=$PDB -v t=$T '/SOLN +[0-9]+ +[0-9]+ .*true/ {k=$3+0; if (min=="" || k<min) min=k} /^ROW/ {row=$0} END {print "FTR pdb="pdb" trial="t" first_true_k="min"  "row}'
done
echo DONE
