#!/bin/bash
# Second-copy placement on the multi-chain structures: oracle (others fixed at
# deposited positions), the fragment with the true orientation injected, and
# the sequential run that fixes what it placed. Three seeds each.
#SBATCH --job-name=chainmode
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-7
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
CELLS=("oracle 3E98 A" "oracle 3E98 B" "oracle 6G9X A" "oracle 6G9X B" "oracle 4BX9 A" "oracle 4BX9 B" "oracle 4BX9 C" "oracle 3VRJ B")

read MODE PDB CH <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for T in 0 1 2; do
  if [ "$MODE" = oracle ]; then
    "$PY" -u alignment_lab/diagnostics/chain_pose_recovery.py --pdb "$PDB" --chain "$CH" --mode oracle --trial $T 2>&1 \
      | grep -v "Warning\|warnings.warn" | grep -A12 "^ROW\|Traceback"
    if [ "$CH" = C ]; then
      "$PY" -u alignment_lab/diagnostics/chain_pose_recovery.py --pdb "$PDB" --chain "$CH" --mode oracle --inject-true --trial $T 2>&1 \
        | grep -v "Warning\|warnings.warn" | grep -A12 "^ROW\|Traceback"
    fi
  else
    "$PY" -u alignment_lab/diagnostics/chain_pose_recovery.py --pdb "$PDB" --mode sequential --trial $T 2>&1 \
      | grep -v "Warning\|warnings.warn" | grep -A12 "^ROW\|^SUMMARY\|Traceback"
  fi
done
echo DONE
