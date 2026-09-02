#!/bin/bash
#SBATCH --job-name=chainv
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-5
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
CELLS=("3E98 B 0" "3E98 B 1" "3E98 B 2" "6G9X B 0" "6G9X B 1" "6G9X B 2")
read PDB CH T <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
"$PY" -u alignment_lab/diagnostics/chain_pose_recovery.py --pdb "$PDB" --chain "$CH" --trial $T 2>&1 \
  | grep -v "Warning\|warnings.warn" | grep -E "^ROW|^CAND|trans[0-9]|^mr:|Traceback" | head -120
echo DONE
