#!/bin/bash
# Single-chain search models: every protein chain of 90+ residues, three seeds.
#SBATCH --job-name=chain
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-14
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
CELLS=("1DAW A" "3E98 A" "3E98 B" "3A5V A" "3VRJ A" "3VRJ B" "1AK5 A" "3K7M X" "3GR5 A" "2DQ6 A" "4BX9 A" "4BX9 B" "4BX9 C" "6G9X A" "6G9X B")
read PDB CH <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for T in 0 1 2; do
  "$PY" -u alignment_lab/diagnostics/chain_pose_recovery.py --pdb "$PDB" --chain "$CH" --trial $T 2>&1 \
    | grep -v "Warning\|warnings.warn" | grep -A12 "^ROW\|Traceback"
done
echo DONE
