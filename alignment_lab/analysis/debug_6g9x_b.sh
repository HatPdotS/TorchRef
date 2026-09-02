#!/bin/bash
#SBATCH --job-name=dbg6g9x
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for C in "6G9X B 0" "3E98 B 0" "3E98 B 1"; do read P CH T <<< "$C"; echo "=== $C"; "$PY" -u alignment_lab/diagnostics/debug_6g9x_b.py --pdb $P --chain $CH --trial $T 2>&1 | grep -v "Warning\|warnings.warn\|copied" | grep -E "^siblings|^deposited|^SOL|^DEPOSITED|^CAND|trans[0-9]|Traceback" -A2; done
echo DONE
