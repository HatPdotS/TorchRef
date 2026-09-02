#!/bin/bash
# The same warm single-process timing on an A100, model and data on the GPU.
#SBATCH --job-name=ptimegpu
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 TORCHREF_DEVICE=cuda
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
"$PY" -u alignment_lab/diagnostics/pipeline_timing.py --threads 8 --device cuda 2>&1 \
  | grep -v "Warning\|warnings.warn" | grep -E "^ROW|^stage|^[0-9]_|^TOTAL|^---|Traceback|Error" | grep -v "^---" 
echo DONE
