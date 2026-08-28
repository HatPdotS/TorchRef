#!/bin/bash
#SBATCH --job-name=etable
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)"
for pdb in 1DAW 3K7M 6G9X 2DQ6; do
  "$PY" -u alignment_lab/analysis/e_table.py --pdb "$pdb" 2>&1 \
    | grep -vE "UserWarning|FutureWarning|^ *from |^ *warnings\.|^Loaded |^LINK |^Wilson outlier|^found non|^FrenchWilson initialized|^  Reflections:|^  Resolution:|^  Space group|^  Centric:|^✓|^Parametrization|^French-Wilson input guard"
  echo
done
echo "rc=$?"
