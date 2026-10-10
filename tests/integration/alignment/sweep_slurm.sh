#!/bin/bash
#SBATCH --job-name=fit_to_data
#SBATCH --output=slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=day

# Usage: sbatch sweep_slurm.sh <PDB_KEY> <SEED>
# E.g.   sbatch sweep_slurm.sh 1AK5 12345

set -euo pipefail

PDB_KEY="${1:?need PDB key}"
SEED="${2:?need seed}"

REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/fix_alignment
PYTHON=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/fix_alignment/.venv/bin/python

cd "$REPO"
export PYTHONPATH="$REPO"
export TORCHREF_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Running fit on PDB=$PDB_KEY seed=$SEED on $(hostname)"
$PYTHON tests/integration/alignment/run_random_pdb_fit.py \
    --pdb "$PDB_KEY" --seed "$SEED" --n-trials 1 --verbose 1
